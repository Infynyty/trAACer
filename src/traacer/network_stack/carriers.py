import numpy as np

def sine_wave(
    frequency: float,
    amplitude: float = 0.5,
    phase: float = 0.0,
    sample_rate: int = 44100,
):
    t = np.linspace(0, 1, sample_rate, endpoint=False)
    signal = amplitude * np.sin(2 * np.pi * frequency * t + phase)
    return signal

def sawtooth_wave(
        period: float,
        amplitude: float = 0.5,
        phase: float = 0.0,
        sample_rate: int = 44100
):
    t = np.linspace(0, 1, sample_rate, endpoint=False)
    signal = amplitude * 2 * ((t / period) - np.floor(0.5 + (t / period))) + phase
    return signal