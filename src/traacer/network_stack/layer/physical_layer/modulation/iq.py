from dataclasses import dataclass

import numpy as np
from numpy import ndarray
import numpy.typing as npt
from scipy.signal import firwin, lfilter

from traacer.network_stack.layer.base import Stage, DataBlock, ComplexArray, AudioSampleBlock, Stream, InT, OutT
from traacer.network_stack.layer.physical_layer.synchronization import PacketAudioSampleBlock


def iq_modulate(
    symbols: npt.NDArray[np.complexfloating],
    frequency: int,
    sample_rate: int
):
    n = np.arange(len(symbols))
    phase = 2 * np.pi * frequency * n / sample_rate

    return symbols.real * np.cos(phase) - symbols.imag * np.sin(phase)

@dataclass(frozen=True, slots=True)
class IQSymbolBlock(DataBlock[ComplexArray]):
    pass

@dataclass
class IQConfig:
    carrier_frequency: int
    sample_rate: int
    lowpass_cutoff: float
    numtaps: int
    carrier_phase_origin: int

class IQSymbolsToAudioSamples(
    Stage[IQSymbolBlock, AudioSampleBlock]
):
    def __init__(self, config: IQConfig):
        self.config = config

    async def process(self, stream: Stream[IQSymbolBlock]) -> Stream[AudioSampleBlock]:
        async for block in stream:
            n = np.arange(len(block.data))
            phase = 2 * np.pi * self.config.carrier_frequency * n / self.config.sample_rate

            data = np.asarray(block.data.real * np.cos(phase) - block.data.imag * np.sin(phase))
            yield AudioSampleBlock(
                data=data,
                is_final=block.is_final,
                metadata=block.metadata
            )

class AudioSamplesToIQSymbols(
    Stage[PacketAudioSampleBlock, IQSymbolBlock]
):
    def __init__(self, config: IQConfig):
        self.config = config

    async def process(self, stream: Stream[PacketAudioSampleBlock]) -> Stream[IQSymbolBlock]:
        async for block in stream:
            n = np.arange(len(block.data))
            phase_n = n - self.config.carrier_phase_origin

            mixer = np.exp(-1j * 2 * np.pi * self.config.carrier_frequency * phase_n / self.config.sample_rate)

            mixed = 2 * block.data * mixer

            taps = firwin(self.config.numtaps, self.config.lowpass_cutoff, fs=self.config.sample_rate)

            pad = (self.config.numtaps - 1) // 2
            mixed_padded = np.pad(mixed, (pad, pad))

            filtered = lfilter(taps, 1.0, mixed_padded)

            data = np.asarray(filtered[2 * pad: 2 * pad + len(block.data)], dtype=np.complex128)

            print("Actual IQ symbols: " + str(data))
            yield IQSymbolBlock(
                data=data,
                is_final=block.is_final,
                metadata=block.metadata
            )

def iq_demodulate(
    signal: npt.NDArray[np.floating],
    carrier_frequency: float,
    sample_rate: float,
    lowpass_cutoff: float,
    numtaps: int = 257,
    carrier_phase_origin: int = 0,
) -> npt.NDArray[np.complexfloating]:
    n = np.arange(len(signal))
    phase_n = n - carrier_phase_origin

    mixer = np.exp(-1j * 2 * np.pi * carrier_frequency * phase_n / sample_rate)

    mixed = 2 * signal * mixer

    taps = firwin(numtaps, lowpass_cutoff, fs=sample_rate)

    pad = (numtaps - 1) // 2
    mixed_padded = np.pad(mixed, (pad, pad))

    filtered = lfilter(taps, 1.0, mixed_padded)

    return filtered[2 * pad: 2 * pad + len(signal)]

