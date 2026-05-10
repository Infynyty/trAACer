import numpy as np
from numpy import ndarray
import numpy.typing as npt
from scipy.signal import firwin, lfilter


def iq_modulate(
    symbols: npt.NDArray[np.complexfloating],
    frequency: int,
    sample_rate: int
):
    n = np.arange(len(symbols))
    phase = 2 * np.pi * frequency * n / sample_rate

    return symbols.real * np.cos(phase) - symbols.imag * np.sin(phase)

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

