import asyncio
import socket
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from ipaddress import ip_address
from pathlib import Path

import qrcode
from cryptography import x509
from cryptography.hazmat._oid import ExtendedKeyUsageOID, NameOID
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI()

BASE_DIR = Path(__file__).resolve().parent
static_dir = BASE_DIR / "static"
static_dir.mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory=static_dir), name="static")


@dataclass
class PcmAudioSession:
    queue: asyncio.Queue[bytes | None] = field(default_factory=asyncio.Queue)
    waiting_for_device: bool = False
    device_ready: bool = False
    stop_requested: bool = False
    upload_active: bool = False
    download_active: bool = False
    sample_rate: int | None = None


pcm_session = PcmAudioSession()


def _get_local_ips() -> list[str]:
    ips = {"127.0.0.1"}
    hostname = socket.gethostname()

    try:
        for info in socket.getaddrinfo(hostname, None, family=socket.AF_INET):
            ip = info[4][0]
            if not ip.startswith("127."):
                ips.add(ip)
    except Exception:
        pass

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            if ip:
                ips.add(ip)
    except Exception:
        pass

    return sorted(ips)


def ensure_dev_cert(cert_path: Path, key_path: Path) -> tuple[Path, Path]:
    if cert_path.exists() and key_path.exists():
        return cert_path, key_path

    cert_path.parent.mkdir(parents=True, exist_ok=True)

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.COUNTRY_NAME, "CH"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Local Dev"),
        x509.NameAttribute(NameOID.COMMON_NAME, "localhost"),
    ])

    san_entries = [x509.DNSName("localhost")]
    for raw_ip in _get_local_ips():
        san_entries.append(x509.IPAddress(ip_address(raw_ip)))

    now = datetime.now(timezone.utc)

    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=5))
        .not_valid_after(now + timedelta(days=365))
        .add_extension(x509.SubjectAlternativeName(san_entries), critical=False)
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                content_commitment=False,
                key_encipherment=True,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=False,
                crl_sign=False,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .add_extension(
            x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]),
            critical=False,
        )
        .sign(private_key=key, algorithm=hashes.SHA256())
    )

    key_path.write_bytes(
        key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )

    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))

    return cert_path, key_path


cert_dir = BASE_DIR / "certs"
cert_path = cert_dir / "dev-cert.pem"
key_path = cert_dir / "dev-key.pem"
cert_path, key_path = ensure_dev_cert(cert_path, key_path)


def _state_json() -> dict[str, object]:
    return {
        "waiting_for_device": pcm_session.waiting_for_device,
        "device_ready": pcm_session.device_ready,
        "stop_requested": pcm_session.stop_requested,
        "upload_active": pcm_session.upload_active,
        "download_active": pcm_session.download_active,
        "sample_rate": pcm_session.sample_rate,
    }


@app.get("/", response_class=HTMLResponse)
async def index():
    html = static_dir.joinpath("index.html").read_text(encoding="utf-8")
    return HTMLResponse(content=html)


@app.get("/api/status")
async def get_status():
    return JSONResponse(_state_json())


@app.post("/api/device-ready")
async def device_ready():
    if pcm_session.waiting_for_device:
        pcm_session.device_ready = True
        pcm_session.stop_requested = False
        return {"ok": True, "message": "Device marked as ready"}

    return {"ok": False, "message": "Server is not waiting for a device"}


@app.get("/api/should-stop")
async def should_stop():
    return {"stop_requested": pcm_session.stop_requested}


@app.post("/api/admin/start-waiting")
async def start_waiting():
    global pcm_session

    await pcm_session.queue.put(None)

    pcm_session = PcmAudioSession(
        waiting_for_device=True,
        device_ready=False,
        stop_requested=False,
        upload_active=False,
        download_active=False,
        sample_rate=None,
    )

    return {"ok": True, "state": _state_json()}


@app.post("/api/admin/request-stop")
async def request_stop():
    pcm_session.stop_requested = True
    await pcm_session.queue.put(None)
    return {"ok": True, "state": _state_json()}


@app.post("/api/admin/reset")
async def reset():
    global pcm_session

    await pcm_session.queue.put(None)
    pcm_session = PcmAudioSession()

    return {"ok": True, "state": _state_json()}


@app.websocket("/api/pcm-upload")
async def pcm_upload(websocket: WebSocket):
    print("Trying to connect")
    await websocket.accept()
    print("Connected")

    if not pcm_session.waiting_for_device:
        await websocket.close(code=1008)
        print("Closed")
        return

    try:
        hello = await websocket.receive_json()
        pcm_session.sample_rate = int(hello["sample_rate"])
        pcm_session.upload_active = True

        while True:
            message = await websocket.receive()

            if message.get("type") == "websocket.disconnect":
                break

            data = message.get("bytes")
            text = message.get("text")

            if data is not None:
                await pcm_session.queue.put(data)

            if text == "stop":
                break

    except WebSocketDisconnect:
        print("Stopped the ws")
        pass

    finally:
        pcm_session.upload_active = False
        pcm_session.waiting_for_device = False
        pcm_session.device_ready = False
        pcm_session.stop_requested = False
        await pcm_session.queue.put(None)


@app.websocket("/api/pcm-download")
async def pcm_download(websocket: WebSocket):
    await websocket.accept()

    pcm_session.download_active = True
    print("Download client connected.")

    try:
        await websocket.send_json(
            {
                "type": "metadata",
                "sample_rate": pcm_session.sample_rate,
                "dtype": "float32",
                "channels": 1,
            }
        )

        while True:
            chunk = await pcm_session.queue.get()

            if chunk is None:
                await websocket.send_json({"type": "stop"})
                break

            await websocket.send_bytes(chunk)

    except WebSocketDisconnect:
        pass

    finally:
        pcm_session.download_active = False