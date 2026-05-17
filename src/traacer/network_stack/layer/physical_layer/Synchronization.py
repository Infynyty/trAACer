from dataclasses import dataclass

import numpy as np

from traacer.network_stack.layer.base import AudioSampleBlock, Stage, Stream, AudioSampleArray, DataBlock


def create_chirp_preamble(
                       start_frequency=2000,
                       end_frequency=8000,
                       duration_in_sec=0.02,
                       sampling_frequency=44100) -> AudioSampleArray:

    N = int(duration_in_sec * sampling_frequency)
    t = np.arange(N) / sampling_frequency

    k = (end_frequency - start_frequency) / duration_in_sec

    chirp = np.cos(2 * np.pi * (start_frequency * t + 0.5 * k * t**2))

    window = np.hanning(N)
    return chirp * window

def create_zadoff_chu_preamble() -> AudioSampleArray:
    N = 839
    u = 25
    t = np.arange(N, dtype=np.float64)

    zadoff_chu = np.exp(-1j * np.pi * u * t * (t + 1) / N)
    window = np.hanning(N).astype(np.float64)

    signal = zadoff_chu * window
    return signal.astype(np.float64)



def find_preamble_start(
        received_signal: AudioSampleArray,
        preamble: AudioSampleArray
) -> int:
    correlation = np.correlate(received_signal, preamble, mode="valid")
    start_index = int(np.argmax(np.abs(correlation)))
    return start_index

class PrependPreamble(Stage[AudioSampleBlock, AudioSampleBlock]):

    def __init__(self, preamble: AudioSampleArray):
        self.preamble = preamble

    async def process(self, stream: Stream[AudioSampleBlock]) -> Stream[AudioSampleBlock]:
        async for block in stream:
            yield AudioSampleBlock(
                data=np.concatenate([self.preamble, block.data]),
                is_final=block.is_final,
                metadata=block.metadata,
            )

@dataclass(frozen=True, slots=True, kw_only=True)
class PacketAudioSampleBlock(DataBlock[AudioSampleArray]):
    packet_index: int
    preamble_start: int
    packet_start: int
    correlation_score: float


class FindPackets(Stage[AudioSampleBlock, PacketAudioSampleBlock]):
    def __init__(
        self,
        preamble: AudioSampleArray,
        packet_num_samples: int,
        threshold: float,
    ):
        self.preamble = preamble
        self.packet_num_samples = packet_num_samples
        self.threshold = threshold
        self.buffer = np.empty(0, dtype=np.float64)
        self.absolute_offset = 0
        self.packet_index = 0

    async def process(
        self,
        stream: Stream[AudioSampleBlock],
    ) -> Stream[PacketAudioSampleBlock]:
        async for block in stream:
            self.buffer = np.concatenate([self.buffer, block.data])

            while True:
                detection = self._find_next_packet()

                if detection is None:
                    break

                preamble_start, packet_start, packet_end, score = detection

                yield PacketAudioSampleBlock(
                    data=self.buffer[packet_start:packet_end],
                    packet_index=self.packet_index,
                    preamble_start=self.absolute_offset + preamble_start,
                    packet_start=self.absolute_offset + packet_start,
                    correlation_score=score,
                    is_final=False,
                )

                self.packet_index += 1
                self.buffer = self.buffer[packet_end:]
                self.absolute_offset += packet_end


    def _normalized_correlation(
            self,
            samples: np.typing.NDArray[np.floating],
            preamble: np.typing.NDArray[np.floating],
    ) -> np.typing.NDArray[np.float64]:
        samples = samples.astype(np.float64)
        preamble = preamble.astype(np.float64)

        corr = np.correlate(samples, preamble, mode="valid")

        window_energy = np.sqrt(
            np.convolve(samples * samples, np.ones(len(preamble)), mode="valid")
        )
        preamble_energy = np.linalg.norm(preamble)

        return corr / (window_energy * preamble_energy + 1e-12)

    def _find_next_packet(self) -> tuple[int, int, int, float] | None:
        if len(self.buffer) < len(self.preamble):
            return None

        score = self._normalized_correlation(self.buffer, self.preamble)
        candidates = np.flatnonzero(score > self.threshold)

        if len(candidates) == 0:
            keep = len(self.preamble) - 1
            if len(self.buffer) > keep:
                drop = len(self.buffer) - keep
                self.buffer = self.buffer[drop:]
                self.absolute_offset += drop
            return None

        first = int(candidates[0])
        radius = len(self.preamble) // 2
        lo = max(0, first - radius)
        hi = min(len(score), first + radius + 1)

        preamble_start = lo + int(np.argmax(score[lo:hi]))
        packet_start = preamble_start + len(self.preamble)
        packet_end = packet_start + self.packet_num_samples

        if packet_end > len(self.buffer):
            if preamble_start > 0:
                self.buffer = self.buffer[preamble_start:]
                self.absolute_offset += preamble_start
            return None

        return (
            preamble_start,
            packet_start,
            packet_end,
            float(score[preamble_start]),
        )