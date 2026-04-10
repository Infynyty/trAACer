import time
from pathlib import Path

import av
import numpy as np
import requests

from traacer.network_stack.layer.device_layer.SDDeviceLayer import SDDeviceLayerSender
from traacer.network_stack.layer.device_layer.WAVDeviceLayer import WAVDeviceLayerSender
from traacer.network_stack.layer.physical_layer.PhysicalLayer import PhysicalLayerReceiver
from traacer.network_stack.packet import PhysicalLayerPacket, DeviceLayerPacket
from traacer.receiver.server import cert_path


class WebserverDeviceLayerSender:

    def __init__(self, server_url: str = "https://127.0.0.1:8000"):
        self.server_url = server_url
        self.sd_sender = SDDeviceLayerSender(44100)
        self.session = requests.Session()
        self.session.verify = cert_path

    def _url(self, path: str) -> str:
        return f"{self.server_url}{path}"

    def _poll_until(
        self,
        predicate,
        timeout_s: float = 60.0,
        interval_s: float = 0.25,
    ):
        deadline = time.monotonic() + timeout_s

        while time.monotonic() < deadline:
            value = predicate()
            if value:
                return value
            time.sleep(interval_s)

        raise TimeoutError(f"Polling timed out after {timeout_s:.1f}s")

    def _get_status(self) -> dict:
        response = self.session.get(self._url("/api/status"), timeout=3)
        response.raise_for_status()
        return response.json()

    def _try_start_waiting(self):
        try:
            response = self.session.post(
                self._url("/api/admin/start-waiting"),
                timeout=3,
            )
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException:
            return None

    def start(self):
        response = self._poll_until(
            lambda: self._try_start_waiting(),
            timeout_s=60.0,
            interval_s=0.25,
        )
        response.raise_for_status()

        self._poll_until(
            lambda: self._get_status().get("waiting_for_device") is True,
            timeout_s=5.0,
        )

        self._poll_until(
            lambda: self._get_status().get("device_ready") is True,
            timeout_s=300.0,
        )

    def stop(self):
        self.session.post(self._url("/api/admin/request-stop"))

    def send_down(self, packet: PhysicalLayerPacket):
        self.sd_sender.send_down(packet)


class WebserverDeviceLayerReceiver:
    def __init__(self, physical_layer_receiver: PhysicalLayerReceiver, recordings_dir: str = "recordings"):
        self.physical_layer_receiver = physical_layer_receiver
        self.recordings_dir = Path(recordings_dir)
        self.last_seen: Path | None = None

    def _latest_file(self) -> Path | None:
        files = [p for p in self.recordings_dir.iterdir() if p.is_file()]
        if not files:
            return None
        return max(files, key=lambda p: p.stat().st_mtime)

    def _poll_until_new_file(
        self,
        timeout_s: float = 300.0,
        interval_s: float = 0.5,
    ) -> Path:
        deadline = time.monotonic() + timeout_s

        while time.monotonic() < deadline:
            path = self._latest_file()
            if path is not None and path != self.last_seen:
                self.last_seen = path
                return path
            time.sleep(interval_s)

        raise TimeoutError(f"Polling timed out after {timeout_s:.1f}s")

    def _load_audio_as_float32(
        self,
        path: Path,
        target_sample_rate: int = 44100,
    ) -> np.ndarray:
        container = av.open(str(path))
        stream = next(s for s in container.streams if s.type == "audio")

        chunks = []
        output_layout = "mono"

        resampler = av.audio.resampler.AudioResampler(
            format="fltp",
            layout=output_layout,
            rate=target_sample_rate,
        )

        for packet in container.demux(stream):
            for frame in packet.decode():
                resampled = resampler.resample(frame)
                if resampled is None:
                    continue

                if not isinstance(resampled, list):
                    resampled = [resampled]

                for out_frame in resampled:
                    arr = out_frame.to_ndarray()
                    if arr.ndim == 2:
                        arr = arr[0]
                    chunks.append(arr.astype(np.float32, copy=False))

        tail = resampler.resample(None)
        if tail:
            if not isinstance(tail, list):
                tail = [tail]
            for out_frame in tail:
                arr = out_frame.to_ndarray()
                if arr.ndim == 2:
                    arr = arr[0]
                chunks.append(arr.astype(np.float32, copy=False))

        container.close()

        if not chunks:
            return np.zeros(0, dtype=np.float32)

        audio = np.concatenate(chunks)

        if audio.size and np.max(np.abs(audio)) > 1.0:
            audio = audio / np.max(np.abs(audio))

        return np.clip(audio, -1.0, 1.0).astype(np.float32, copy=False)

    def send_up(self):
        path = self._poll_until_new_file()
        self.physical_layer_receiver.send_up(DeviceLayerPacket(self._load_audio_as_float32(path)))