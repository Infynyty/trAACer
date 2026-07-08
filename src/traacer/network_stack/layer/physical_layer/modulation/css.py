from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray
from scipy.signal import spectrogram

from traacer.network_stack.layer.physical_layer.Synchronization import create_chirp_preamble


def css_modulate(
        data: ndarray,
        samples_per_second: int,
        start_frequency: int,
        end_frequency: int,
        symbol_duration: float,
        number_of_symbols: int
):
    if samples_per_second < 0:
        raise Exception("Samples per second must be positive.")
    if end_frequency < start_frequency:
        raise Exception("Stopping frequency must be larger than starting frequency.")
    if symbol_duration < 0:
        raise Exception("Symbol duration must be positive")
    if number_of_symbols < 0 and (number_of_symbols & (number_of_symbols - 1)) != 0:
        raise Exception("Number of symbols must be a power of 2")

    bits_per_symbol = int(np.log2(number_of_symbols))

    if len(data) % bits_per_symbol != 0:
        raise Exception("The length of the input data is not divisible by the bits per symbol.")

    number_of_generated_symbols = int(len(data) / bits_per_symbol)

    chirp = create_chirp(samples_per_second, start_frequency, end_frequency, symbol_duration)

    shifts = (np.arange(number_of_symbols) * len(chirp)) // number_of_symbols
    indices = (np.arange(len(chirp))[None, :] - shifts[:, None]) % len(chirp)

    symbol_table = chirp[indices]

    signal = []

    for i in range(number_of_generated_symbols):
        symbol_index = 0
        for j in range(bits_per_symbol):
            symbol_index = (symbol_index << 1) | int(data[i * bits_per_symbol + j])
        signal.append(symbol_table[symbol_index])

    return np.concat(signal)


def create_chirp(samples_per_second: int, start_frequency: int, end_frequency: int, symbol_duration: float) -> Any:
    N = int(symbol_duration * samples_per_second)
    t = np.arange(N) / samples_per_second

    k = (end_frequency - start_frequency) / symbol_duration

    chirp = np.cos(2 * np.pi * (start_frequency * t + 0.5 * k * t ** 2))
    return chirp


def css_detect_signal(
        signal,
        base_chirp
):
    R = np.fft.fft(signal)
    C = np.fft.fft(base_chirp)
    z = np.fft.ifft(R * np.conj(C))
    k_hat = int(np.argmax(np.abs(z)))
    return k_hat, z

def css_demodulate(
        signal: ndarray,
        samples_per_second: int,
        start_frequency: int,
        end_frequency: int,
        symbol_duration: float,
        number_of_symbols: int
):
    signal = np.asarray(signal)
    base_chirp = create_chirp(samples_per_second, start_frequency, end_frequency, symbol_duration)
    base_chirp = np.asarray(base_chirp)

    symbol_len = len(base_chirp)
    n_symbols = len(signal) // symbol_len

    signal = signal[:n_symbols * symbol_len]
    symbols = signal.reshape(n_symbols, symbol_len)

    detected = np.empty(n_symbols, dtype=int)

    step = symbol_len // number_of_symbols

    for i, sym in enumerate(symbols):
        k_hat, _ = css_detect_signal(sym, base_chirp)
        detected[i] = int(np.round(k_hat / step)) % number_of_symbols

    return symbols_to_bits(detected, number_of_symbols)

def symbols_to_bits(symbols, number_of_symbols):
    symbols = np.asarray(symbols)

    bits_per_symbol = int(np.log2(number_of_symbols))
    assert 2 ** bits_per_symbol == number_of_symbols

    out = np.zeros(len(symbols) * bits_per_symbol, dtype=int)

    for i, s in enumerate(symbols):
        for b in range(bits_per_symbol):
            out[i * bits_per_symbol + (bits_per_symbol - 1 - b)] = (s >> b) & 1

    return out