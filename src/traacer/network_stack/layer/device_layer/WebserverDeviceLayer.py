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

from traacer.receiver.server import cert_path

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