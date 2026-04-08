from pathlib import Path
from datetime import datetime

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