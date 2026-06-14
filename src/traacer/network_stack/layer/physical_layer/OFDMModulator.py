from dataclasses import dataclass
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np

from traacer.metrics.bit_comparator import bit_queue, compare_bits
from traacer.metrics.plot import plot_signal, plot_constellation
from traacer.network_stack.layer.base import StreamingProcessor, BitBlock, DataBlock, ComplexArray, InT, OutT, Stage, \
    Stream
from traacer.network_stack.layer.physical_layer.IQModulator import iq_modulate, iq_demodulate, IQSymbolBlock


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

def qam_gray_lookup_table(mu):
    if mu % 2 != 0:
        raise ValueError("n must be even for square QAM")

    bits_per_axis = mu // 2
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

@dataclass(frozen=True, slots=True)
class QAMSymbolBlock(DataBlock[ComplexArray]):
    pass

class BitsToQAMSymbols(
    StreamingProcessor[BitBlock, QAMSymbolBlock]
):
    def __init__(self, mu: int):
        if mu % 2 != 0:
            raise ValueError("n must be even for square QAM")
        self.mu = mu
        self.lut = qam_gray_lookup_table(mu)
        self.buffer = np.empty(0, dtype=np.uint8)

    def push(self, block: BitBlock) -> Iterable[QAMSymbolBlock]:
        bits = np.asarray(block.data, dtype=np.uint8)

        if self.buffer.size > 0:
            bits = np.concatenate([self.buffer, bits])

        usable_len = (len(bits) // self.mu) * self.mu
        usable_bits = bits[:usable_len]
        self.buffer = bits[usable_len:]

        if usable_len == 0:
            return []

        symbols = []
        for i in range(0, usable_len, self.mu):
            chunk = usable_bits[i:i + self.mu]
            symbols.append(self.lut[tuple(chunk.tolist())])

        return [
            QAMSymbolBlock(
                data=np.asarray(symbols, dtype=np.complex128),
                is_final=False,
                metadata=block.metadata,
            )
        ]

class QAMSymbolsToBits(
    Stage[QAMSymbolBlock, BitBlock]
):
    def __init__(self, mu: int):
        if mu % 2 != 0:
            raise ValueError("mu must be even for square QAM")
        self.mu = mu
        self.lut = qam_gray_lookup_table(mu)
        self.bit_patterns = list(self.lut.keys())
        self.constellation = np.array([self.lut[bits] for bits in self.bit_patterns], dtype=np.complex128)
        self.bit_pattern_array = np.asarray(
            self.bit_patterns,
            dtype=np.uint8,
        )

    async def process(self, stream: Stream[QAMSymbolBlock]) -> Stream[BitBlock]:
        async for block in stream:
            distances = np.abs(block.data[:, None] - self.constellation[None, :])
            nearest_indices = np.argmin(distances, axis=1)

            output_bits = self.bit_pattern_array[nearest_indices].reshape(-1)

            bit_queue.put(output_bits)
            compare_bits()

            yield BitBlock(
                data=output_bits.astype(np.uint8),
                is_final=block.is_final,
                metadata=block.metadata,
            )



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


@dataclass(frozen=True, slots=True)
class OFDMFrameBlock(DataBlock[ComplexArray]):
    pass

@dataclass
class OFDMConfig:
    center_frequency: int
    sample_rate: int
    cp_len: int
    n_fft: int
    pilot_spacing: int
    pilot_value: complex
    num_positive_subcarriers: int

    def get_samples_per_ofdm_frame(self) -> int:
        return self.cp_len + self.n_fft

class QAMSymbolsToOFDMFrame(
    Stage[QAMSymbolBlock, IQSymbolBlock]
):
    def __init__(self, config: OFDMConfig):
        if config.cp_len < 0 or config.cp_len > config.n_fft:
            raise ValueError("cp_len must be between 0 and frame_len")

        if config.pilot_spacing <= 0:
            raise ValueError("pilot_spacing must be positive")
        self.config = config

    async def process(self, stream: Stream[QAMSymbolBlock]) -> Stream[IQSymbolBlock]:
        async for block in stream:
            subcarriers = np.zeros(self.config.n_fft, dtype=np.complex128)

            used_indices = get_used_indices(self.config.n_fft, self.config.num_positive_subcarriers)
            pilot_indices = used_indices[::self.config.pilot_spacing]
            data_indices = np.setdiff1d(used_indices, pilot_indices)

            if len(block.data) > len(data_indices):
                raise ValueError(
                    f"Too many symbols for OFDM frame, got {len(block.data)}, "
                    f"but only {len(data_indices)} data indices are available."
                )

            data_indices = data_indices[:len(block.data)]

            subcarriers[pilot_indices] = self.config.pilot_value
            subcarriers[data_indices] = block.data

            time_symbols = np.fft.ifft(subcarriers, n=self.config.n_fft)

            cyclic_prefix = time_symbols[-self.config.cp_len:] if self.config.cp_len > 0 else np.array([], dtype=time_symbols.dtype)
            time_symbols_with_cp = np.concatenate([cyclic_prefix, time_symbols])

            yield IQSymbolBlock(
                data=np.array(time_symbols_with_cp, dtype=np.complex128),
                is_final=block.is_final,
                metadata=block.metadata
            )


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

class IQSymbolsToQAMSymbols(
    StreamingProcessor[IQSymbolBlock, QAMSymbolBlock]
):

    def __init__(self, config: OFDMConfig):
        self.config = config
        self.buffer = np.empty(0, dtype=np.complex128)

    def equalize_with_pilots(
            self,
            qam_symbols: np.typing.NDArray[np.complexfloating],
            pilots: np.typing.NDArray[np.complexfloating],
            pilot_indices: np.typing.NDArray[np.integer],
            data_indices: np.typing.NDArray[np.integer],
    ) -> np.typing.NDArray[np.complexfloating]:
        h_pilots = pilots / self.config.pilot_value
        h_data = np.empty(len(data_indices), dtype=np.complex128)

        positive_pilot_mask = pilot_indices < self.config.n_fft // 2
        negative_pilot_mask = pilot_indices > self.config.n_fft // 2

        positive_data_mask = data_indices < self.config.n_fft // 2
        negative_data_mask = data_indices > self.config.n_fft // 2

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

    def ofdm_demodulate_frame(
            self,
            baseband: np.typing.NDArray[np.complexfloating],
    ) -> np.typing.NDArray[np.complexfloating]:
        frame_len = self.config.cp_len + self.config.n_fft

        if len(baseband) < frame_len:
            raise ValueError("baseband does not contain a complete OFDM frame")

        symbol_with_cp = baseband[:frame_len]
        symbol = symbol_with_cp[
            self.config.cp_len: self.config.cp_len + self.config.n_fft
        ]

        received_symbols = np.fft.fft(symbol)

        used_indices = get_used_indices(
            self.config.n_fft,
            self.config.num_positive_subcarriers,
        )
        pilot_indices = used_indices[:: self.config.pilot_spacing]
        data_indices = np.setdiff1d(used_indices, pilot_indices)

        pilots = received_symbols[pilot_indices]
        qam_symbols = received_symbols[data_indices]

        qam_symbols_equalized = self.equalize_with_pilots(
            qam_symbols=qam_symbols,
            pilots=pilots,
            pilot_indices=pilot_indices,
            data_indices=data_indices,
        )

        return qam_symbols_equalized.astype(np.complex128)

    def push(self, block: IQSymbolBlock) -> Iterable[QAMSymbolBlock]:
        samples = np.asarray(block.data, dtype=np.complex128)

        if len(samples) > 0:
            self.buffer = np.concatenate([self.buffer, samples])

        frame_len = self.config.cp_len + self.config.n_fft

        while len(self.buffer) >= frame_len:
            frame_samples = self.buffer[:frame_len]
            self.buffer = self.buffer[frame_len:]

            qam_symbols = self.ofdm_demodulate_frame(frame_samples)

            yield QAMSymbolBlock(
                data=qam_symbols,
                is_final=block.is_final and len(self.buffer) < frame_len,
                metadata=block.metadata,
            )
        self.buffer = np.empty(0, dtype=np.complex128)