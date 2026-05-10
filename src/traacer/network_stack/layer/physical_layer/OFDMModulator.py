import matplotlib.pyplot as plt
import numpy as np

from traacer.metrics.plot import plot_signal, plot_constellation
from traacer.network_stack.layer.physical_layer.IQModulator import iq_modulate, iq_demodulate


def to_qam_symbol(mu: int):
    levels = np.arange(-mu + 1, mu, 2)
    symbols = []
    for i in levels:
        for q in levels:
            symbols.append(i + 1j * q)

    symbols = np.array(symbols, dtype=complex)
    symbols /= np.sqrt(np.mean(np.abs(symbols) ** 2))
    return symbols

def gray_code(k):
    return k ^ (k >> 1)


def qam_gray_lookup_table(n):
    if n % 2 != 0:
        raise ValueError("n must be even for square QAM")

    bits_per_axis = n // 2
    levels_per_axis = 2 ** bits_per_axis

    levels = np.arange(-(levels_per_axis - 1), levels_per_axis, 2)

    table = {}

    for i_idx in range(levels_per_axis):
        for q_idx in range(levels_per_axis):
            i_gray = i_idx ^ (i_idx >> 1)
            q_gray = q_idx ^ (q_idx >> 1)

            i_bits = tuple(map(int, format(i_gray, f"0{bits_per_axis}b")))
            q_bits = tuple(map(int, format(q_gray, f"0{bits_per_axis}b")))

            bits = i_bits + q_bits
            symbol = levels[i_idx] + 1j * levels[q_idx]

            table[bits] = symbol

    avg_energy = np.mean(np.abs(list(table.values())) ** 2)
    scale = np.sqrt(avg_energy)

    for bits in table:
        table[bits] /= scale

    return table

def bits_to_qam(bits, n):
    if len(bits) % n != 0:
        raise ValueError("Bitstream length must be divisible by n")

    lut = qam_gray_lookup_table(n)
    symbols = []
    for i in range(0, len(bits), n):
        chunk = bits[i:i + n]
        key = tuple(chunk.tolist())
        symbols.append(lut[key])

    return np.array(symbols)

def qam_to_bits(symbols, n):
    lut = qam_gray_lookup_table(n)

    bit_patterns = list(lut.keys())
    constellation = np.array([lut[bits] for bits in bit_patterns], dtype=np.complex128)

    output_bits = []

    for symbol in symbols:
        distances = np.abs(symbol - constellation)
        nearest_index = int(np.argmin(distances))
        output_bits.extend(bit_patterns[nearest_index])

    return np.array(output_bits, dtype=np.uint8)

def get_used_indices(n_fft: int, num_positive_subcarriers: int):
    return np.r_[
        np.arange(1, num_positive_subcarriers + 1),
        np.arange(n_fft - num_positive_subcarriers, n_fft),
    ]

def ofdm_modulate(
        bits: np.typing.NDArray[np.uint8],
        qam_mu: int,
        center_frequency: int,
        sampling_rate: int,
        cp_len: int,
        n_fft: int,
        pilot_spacing: int,
        pilot_value: complex,
    num_positive_subcarriers: int

):
    if cp_len < 0 or cp_len > n_fft:
        raise ValueError("cp_len must be between 0 and frame_len")

    if pilot_spacing <= 0:
        raise ValueError("pilot_spacing must be positive")

    qam_symbols = bits_to_qam(bits, qam_mu)

    subcarriers = np.zeros(n_fft, dtype=np.complex128)

    used_indices = get_used_indices(n_fft, num_positive_subcarriers)
    pilot_indices = used_indices[::pilot_spacing]
    data_indices = np.setdiff1d(used_indices, pilot_indices)

    if len(qam_symbols) > len(data_indices):
        raise ValueError(
            f"Too many symbols for OFDM frame, got {len(qam_symbols)}, "
            f"but only {len(data_indices)} data indices are available."
        )

    data_indices = data_indices[:len(qam_symbols)]

    subcarriers[pilot_indices] = pilot_value
    subcarriers[data_indices] = qam_symbols

    time_symbols = np.fft.ifft(subcarriers, n=n_fft)

    cyclic_prefix = time_symbols[-cp_len:] if cp_len > 0 else np.array([], dtype=time_symbols.dtype)
    time_symbols_with_cp = np.concatenate([cyclic_prefix, time_symbols])

    return iq_modulate(time_symbols_with_cp, center_frequency, sampling_rate)

def equalize_with_pilots(qam_symbols, pilots, pilot_indices, data_indices, pilot_value, n_fft):
    h_pilots = pilots / pilot_value
    h_data = np.empty(len(data_indices), dtype=np.complex128)

    positive_pilot_mask = pilot_indices < n_fft // 2
    negative_pilot_mask = pilot_indices > n_fft // 2

    positive_data_mask = data_indices < n_fft // 2
    negative_data_mask = data_indices > n_fft // 2

    for data_mask, pilot_mask in [
        (positive_data_mask, positive_pilot_mask),
        (negative_data_mask, negative_pilot_mask),
    ]:
        x_data = data_indices[data_mask]
        x_pilot = pilot_indices[pilot_mask]
        h_pilot = h_pilots[pilot_mask]

        if len(x_data) == 0:
            continue

        order = np.argsort(x_pilot)
        x_pilot = x_pilot[order]
        h_pilot = h_pilot[order]

        h_real = np.interp(x_data, x_pilot, np.real(h_pilot))
        h_imag = np.interp(x_data, x_pilot, np.imag(h_pilot))

        h_data[data_mask] = h_real + 1j * h_imag

    return qam_symbols / h_data

def ofdm_demodulate(
    baseband: np.typing.NDArray[np.complexfloating],
    n_fft: int,
    qam_mu: int,
    cp_len: int,
    pilot_spacing: int,
    pilot_value: complex,
    num_positive_subcarriers: int,
):
    symbol_with_cp = baseband[: cp_len + n_fft]
    symbol = symbol_with_cp[cp_len : cp_len + n_fft]

    received_symbols = np.fft.fft(symbol)

    used_indices = get_used_indices(n_fft, num_positive_subcarriers)
    pilot_indices = used_indices[::pilot_spacing]
    data_indices = np.setdiff1d(used_indices, pilot_indices)

    pilots = received_symbols[pilot_indices]
    qam_symbols = received_symbols[data_indices]

    qam_symbols_equalized = equalize_with_pilots(
        qam_symbols=qam_symbols,
        pilots=pilots,
        pilot_indices=pilot_indices,
        data_indices=data_indices,
        pilot_value=pilot_value,
        n_fft=n_fft,
    )

    bits = qam_to_bits(qam_symbols_equalized, qam_mu)
    return bits

