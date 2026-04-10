import socket
from ipaddress import ip_address
from pathlib import Path
from datetime import datetime, timezone, timedelta

from cryptography import x509
from cryptography.hazmat._oid import NameOID, ExtendedKeyUsageOID
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI()

state = {
    "waiting_for_device": False,
    "device_ready": False,
    "stop_requested": False,
}

BASE_DIR = Path(__file__).resolve().parent

recordings_dir = BASE_DIR / "recordings"
recordings_dir.mkdir(parents=True, exist_ok=True)

static_dir = BASE_DIR / "static"

app.mount("/static", StaticFiles(directory=static_dir), name="static")



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

@app.get("/", response_class=HTMLResponse)
async def index():
    html = Path(f"{static_dir}/index.html").read_text(encoding="utf-8")
    return HTMLResponse(content=html)


@app.get("/api/status")
async def get_status():
    return JSONResponse(
        {
            "waiting_for_device": state["waiting_for_device"],
            "device_ready": state["device_ready"],
            "stop_requested": state["stop_requested"],
        }
    )


@app.post("/api/device-ready")
async def device_ready():
    if state["waiting_for_device"]:
        state["device_ready"] = True
        state["stop_requested"] = False
        return {"ok": True, "message": "Device marked as ready"}
    return {"ok": False, "message": "Server is not waiting for a device"}


@app.get("/api/should-stop")
async def should_stop():
    return {"stop_requested": state["stop_requested"]}


@app.post("/api/upload-recording")
async def upload_recording(file: UploadFile = File(...)):
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
    suffix = Path(file.filename or "recording.webm").suffix or ".webm"
    target = recordings_dir / f"recording_{timestamp}{suffix}"

    content = await file.read()
    target.write_bytes(content)

    state["waiting_for_device"] = False
    state["device_ready"] = False
    state["stop_requested"] = False

    return {
        "ok": True,
        "filename": str(target),
        "size": len(content),
    }


@app.post("/api/admin/start-waiting")
async def start_waiting():
    state["waiting_for_device"] = True
    state["device_ready"] = False
    state["stop_requested"] = False
    return {"ok": True, "state": state}


@app.post("/api/admin/request-stop")
async def request_stop():
    state["stop_requested"] = True
    return {"ok": True, "state": state}


@app.post("/api/admin/reset")
async def reset():
    state["waiting_for_device"] = False
    state["device_ready"] = False
    state["stop_requested"] = False
    return {"ok": True, "state": state}