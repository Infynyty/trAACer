import asyncio
import json
import random
import ssl
import time
from pathlib import Path
from typing import Callable, Any

import av
import numpy as np
import requests
import websockets
from websockets import InvalidHandshake, InvalidStatus, ConnectionClosed, WebSocketException

from traacer.network_stack.layer.base import AudioSampleBlock, Source, AudioSampleArray, Stream, Sink
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


import asyncio
import time
from collections.abc import Callable
from typing import Any

import numpy as np
import requests
import sounddevice as sd



class WebserverDeviceLayerSenderSink(Sink[AudioSampleBlock]):
    def __init__(
        self,
        server_url: str = "https://127.0.0.1:8000",
        sample_rate: int = 44100,
        timeout_s: float = 60.0,
        interval_s: float = 0.25,
        cert_path: str | Path = "certs/dev-cert.pem",
    ):
        self.server_url = server_url.rstrip("/")
        self.sample_rate = sample_rate
        self.timeout_s = timeout_s
        self.interval_s = interval_s
        self.cert_path = Path(cert_path)

        self.session = requests.Session()
        self.session.verify = str(self.cert_path)

    def _url(self, path: str) -> str:
        return f"{self.server_url}{path}"

    def _poll_until(
        self,
        predicate: Callable[[], Any],
        timeout_s: float,
        interval_s: float | None = None,
    ) -> Any:
        deadline = time.monotonic() + timeout_s
        interval = self.interval_s if interval_s is None else interval_s

        while time.monotonic() < deadline:
            value = predicate()
            if value:
                return value
            time.sleep(interval)

        raise TimeoutError(f"Polling timed out after {timeout_s:.1f}s")

    def _get_status(self) -> dict[str, Any]:
        response = self.session.get(self._url("/api/status"), timeout=3)
        response.raise_for_status()
        return response.json()

    def _try_start_waiting(self) -> requests.Response | None:
        try:
            response = self.session.post(
                self._url("/api/admin/start-waiting"),
                timeout=3,
            )
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException:
            return None

    def _start_blocking(self) -> None:
        response = self._poll_until(
            lambda: self._try_start_waiting(),
            timeout_s=self.timeout_s,
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

        self._poll_until(
            lambda: self._get_status().get("download_active") is True,
            timeout_s=30.0,
        )

    def _stop_blocking(self) -> None:
        response = self.session.post(
            self._url("/api/admin/request-stop"),
            timeout=3,
        )
        response.raise_for_status()

    @staticmethod
    def _prepare_audio(data: np.ndarray) -> np.ndarray:
        audio = np.asarray(data)

        if audio.ndim == 1:
            audio = audio[:, None]

        if audio.ndim != 2:
            raise ValueError(f"Expected 1D or 2D audio array, got shape {audio.shape}")

        audio = np.clip(audio, -1.0, 1.0)
        return audio.astype(np.float32, copy=False)

    async def consume(self, stream: Stream[AudioSampleBlock]) -> None:
        await asyncio.to_thread(self._start_blocking)

        try:
            with sd.OutputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
            ) as output_stream:
                async for packet in stream:
                    audio = self._prepare_audio(packet.data)
                    await asyncio.to_thread(output_stream.write, audio)

                    if packet.is_final:
                        break

        finally:
            pass
            # await asyncio.to_thread(self._stop_blocking)

class WebserverDeviceLayerReceiverSource(Source[AudioSampleBlock]):
    def __init__(
        self,
        server_url: str = "https://127.0.0.1:8000",
        target_sample_rate: int = 44100,
        cert_path: str | Path = "certs/dev-cert.pem",
    ):
        self.server_url = server_url.rstrip("/")
        self.target_sample_rate = target_sample_rate
        self.cert_path = Path(cert_path)

    def _ws_url(self, path: str) -> str:
        if self.server_url.startswith("https://"):
            base = "wss://" + self.server_url.removeprefix("https://")
        elif self.server_url.startswith("http://"):
            base = "ws://" + self.server_url.removeprefix("http://")
        else:
            raise ValueError(f"Unsupported server URL: {self.server_url}")

        return f"{base}{path}"

    def _ssl_context(self) -> ssl.SSLContext | None:
        if not self.server_url.startswith("https://"):
            return None

        return ssl.create_default_context(cafile=str(self.cert_path))

    async def stream(self) -> Stream[AudioSampleBlock]:
        ws_url = self._ws_url("/api/pcm-download")
        ssl_context = self._ssl_context()

        metadata: dict[str, Any] = {
            "sample_rate": self.target_sample_rate,
            "dtype": "float32",
            "channels": 1,
            "source": "webserver",
        }

        reconnect_attempt = 0

        while True:
            try:
                async with websockets.connect(
                        ws_url,
                        ssl=ssl_context,
                        open_timeout=2.0,
                        ping_interval=20.0,
                        ping_timeout=20.0,
                        close_timeout=2.0,
                        max_size=None,
                ) as websocket:
                    reconnect_attempt = 0

                    while True:
                        message = await websocket.recv()

                        if isinstance(message, str):
                            payload = json.loads(message)

                            if payload.get("type") == "metadata":
                                sample_rate = payload.get("sample_rate")
                                if sample_rate is not None:
                                    metadata["sample_rate"] = int(sample_rate)

                                metadata["dtype"] = payload.get("dtype", "float32")
                                metadata["channels"] = payload.get("channels", 1)
                                continue

                            if payload.get("type") == "stop":
                                yield AudioSampleBlock(
                                    data=np.zeros(0, dtype=np.float64),
                                    is_final=True,
                                    metadata=dict(metadata),
                                )
                                return

                            continue

                        audio = np.frombuffer(message, dtype=np.float32).astype(
                            np.float64,
                            copy=False,
                        )

                        yield AudioSampleBlock(
                            data=audio,
                            is_final=False,
                            metadata=dict(metadata),
                        )

            except asyncio.CancelledError:
                raise

            except (
                    OSError,
                    TimeoutError,
                    ConnectionRefusedError,
                    ConnectionResetError,
                    InvalidHandshake,
                    InvalidStatus,
                    ConnectionClosed,
                    WebSocketException,
            ) as exc:
                reconnect_attempt += 1

                delay = min(10.0, 0.25 * 2 ** (reconnect_attempt - 1))
                delay += random.uniform(0.0, 0.25)

                print(
                    f"WebSocket receive connection failed: "
                    f"{type(exc).__name__}: {exc}. "
                    f"Retrying in {delay:.2f}s"
                )

                await asyncio.sleep(delay)