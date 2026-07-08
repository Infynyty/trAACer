from __future__ import annotations
import wave
from wave import Wave_write, Wave_read

import numpy as np

from traacer.network_stack.layer.base import Sink, Stream, Source, BitBlock, AudioSampleBlock

import wave
from pathlib import Path
from wave import Wave_read, Wave_write

import numpy as np


class WAVSink(Sink[BitBlock]):
    """
    Device layer sink that writes incoming audio sample blocks to a WAV file.

    Input samples may be floating point in [-1, 1] or int16 PCM.
    The WAV file is written as mono 16-bit PCM.
    """

    def __init__(self, path: str | Path, sample_rate: int = 44_100):
        self.path = Path(path)
        self.sample_rate = sample_rate
        self.writer: Wave_write | None = None

    async def consume(self, stream: Stream[BitBlock]) -> None:
        with wave.open(str(self.path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)

            self.writer = wf

            async for packet in stream:
                data = self._to_int16_pcm(packet.data)
                wf.writeframes(data.tobytes())

            self.writer = None

    @staticmethod
    def _to_int16_pcm(data: np.ndarray) -> np.ndarray:
        if np.issubdtype(data.dtype, np.floating):
            data = np.clip(data, -1.0, 1.0)
            return (data * 32767).astype(np.int16)

        if data.dtype == np.int16:
            return data

        raise ValueError(
            f"Unsupported WAV sample dtype {data.dtype}. "
            "Expected floating point samples or int16 PCM."
        )

class WAVSource(Source[AudioSampleBlock]):
    """
    Device layer source that reads a mono 16-bit WAV file and emits one
    DeviceLayerPacket containing normalized float64 samples in [-1, 1].
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.reader: Wave_read | None = None

    async def stream(self) -> Stream[AudioSampleBlock]:
        with wave.open(str(self.path), "rb") as wf:
            self.reader = wf

            if wf.getnchannels() != 1:
                raise ValueError("Only mono WAV supported.")

            if wf.getsampwidth() != 2:
                raise ValueError("Only 16-bit WAV supported.")

            n_frames = wf.getnframes()
            raw_bytes = wf.readframes(n_frames)

            if len(raw_bytes) == 0:
                return

            data = np.frombuffer(raw_bytes, dtype=np.int16)
            data = data.astype(np.float32) / 32767.0

            yield AudioSampleBlock(data=data, is_final=True)

            self.reader = None
