import numpy as np

from traacer.metrics.plot import plot_signal, plot_signal_with_signal_boundaries
from traacer.network_stack.layer.device_layer.DeviceLayer import DeviceLayerSender
from traacer.network_stack.packet import LinkLayerPacket, PhysicalLayerPacket
from traacer.network_stack.layer.physical_layer.PhysicalLayerUtil import bytes_to_bits


class PSKPhysicalLayerSender:

    def __init__(self, device_layer_sender: DeviceLayerSender, sampling_rate = 44100, symbol_period = 0.01, carrier_frequency = 1000):
        self.sampling_rate = sampling_rate
        self.device_layer_sender = device_layer_sender
        self.symbol_period = symbol_period
        self.carrier_frequency = carrier_frequency



    def encode_bits(self, data):
        fs = self.sampling_rate
        T = self.symbol_period
        f = self.carrier_frequency

        samples_per_symbol = int(fs * T)

        t = np.arange(0, T * len(data), 1 / fs)
        carrier = np.cos(2 * np.pi * f * t)

        symbols = 2 * data - 1
        symbol_stream = np.repeat(symbols, samples_per_symbol)

        return carrier * symbol_stream

    def send_down(self, packet: LinkLayerPacket):
        bits = np.fromiter(
            bytes_to_bits(packet.data.tobytes()),
            dtype=np.int8,
        )
        data = self.encode_bits(bits)
        plot_signal_with_signal_boundaries(data, self.symbol_period * self.sampling_rate, 5000)
        self.device_layer_sender.send_down(PhysicalLayerPacket(data))


from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class PSKConfig:
    sample_rate: int
    symbol_rate: float
    carrier_frequency: float
    order: int = 4
    amplitude: float = 0.8
    ramp_duration: float = 0.002
    continuous_phase: bool = True
    phase_offset: float = 0.0

    @property
    def samples_per_symbol(self) -> int:
        return int(round(self.sample_rate / self.symbol_rate))

    @property
    def ramp_samples(self) -> int:
        return min(
            int(round(self.ramp_duration * self.sample_rate)),
            self.samples_per_symbol // 2,
        )

    @property
    def bits_per_symbol(self) -> int:
        return int(math.log2(self.order))

    @property
    def symbol_phases(self) -> np.ndarray:
        return (
            self.phase_offset
            + 2.0 * np.pi * np.arange(self.order, dtype=np.float64) / self.order
        )

class BitsToPSKAudioSamples(Stage[BitBlock, AudioSampleBlock]):
    def __init__(self, config: PSKConfig):
        self.config = config
        self.carrier_phase = 0.0
        self.bit_buffer = np.empty(0, dtype=np.uint8)

        self._validate_config()

    async def process(
        self,
        stream: Stream[BitBlock],
    ) -> Stream[AudioSampleBlock]:
        async for block in stream:
            bits = np.asarray(block.data, dtype=np.uint8)

            if np.any((bits != 0) & (bits != 1)):
                raise ValueError("PSK input bits must only contain 0 and 1")

            bits = np.concatenate([self.bit_buffer, bits])

            bps = self.config.bits_per_symbol
            num_complete_symbols = len(bits) // bps
            num_used_bits = num_complete_symbols * bps

            used_bits = bits[:num_used_bits]
            self.bit_buffer = bits[num_used_bits:]

            if block.is_final and len(self.bit_buffer) > 0:
                raise ValueError(
                    "Final PSK bit block does not contain a whole number "
                    f"of {bps}-bit symbols"
                )

            samples = self._modulate_bits(used_bits)

            yield AudioSampleBlock(
                data=samples,
                is_final=block.is_final,
                metadata=block.metadata,
            )

    def _validate_config(self) -> None:
        if self.config.samples_per_symbol <= 0:
            raise ValueError("samples_per_symbol must be positive")

        if self.config.amplitude < 0:
            raise ValueError("amplitude must be non-negative")

        if self.config.order < 2:
            raise ValueError("order must be at least 2")

        if self.config.order & (self.config.order - 1) != 0:
            raise ValueError("order must be a power of two")

    def _modulate_bits(self, bits: np.ndarray) -> np.ndarray:
        if len(bits) == 0:
            return np.empty(0, dtype=np.float64)

        symbol_indices = self._bits_to_symbol_indices(bits)

        output = np.empty(
            len(symbol_indices) * self.config.samples_per_symbol,
            dtype=np.float64,
        )

        offset = 0

        for symbol_index in symbol_indices:
            symbol = self._modulate_symbol(int(symbol_index))
            output[offset:offset + self.config.samples_per_symbol] = symbol
            offset += self.config.samples_per_symbol

        return output

    def _bits_to_symbol_indices(self, bits: np.ndarray) -> np.ndarray:
        bps = self.config.bits_per_symbol

        if len(bits) % bps != 0:
            raise ValueError(
                f"Number of bits must be divisible by bits_per_symbol={bps}"
            )

        bit_groups = bits.reshape(-1, bps)

        powers = 2 ** np.arange(bps - 1, -1, -1, dtype=np.uint8)
        return bit_groups @ powers

    def _modulate_symbol(self, symbol_index: int) -> np.ndarray:
        if not 0 <= symbol_index < self.config.order:
            raise ValueError(
                f"PSK symbol index must be in [0, {self.config.order - 1}], "
                f"got {symbol_index}"
            )

        n = self.config.samples_per_symbol
        t = np.arange(n, dtype=np.float64) / self.config.sample_rate

        carrier_phase = (
            self.carrier_phase if self.config.continuous_phase else 0.0
        )

        symbol_phase = self.config.symbol_phases[symbol_index]

        symbol = self.config.amplitude * np.sin(
            2.0 * np.pi * self.config.carrier_frequency * t
            + carrier_phase
            + symbol_phase
        )

        if self.config.continuous_phase:
            self.carrier_phase = (
                carrier_phase
                + 2.0
                * np.pi
                * self.config.carrier_frequency
                * n
                / self.config.sample_rate
            ) % (2.0 * np.pi)

        symbol *= self._symbol_window(n)

        return symbol

    def _symbol_window(self, n: int) -> np.ndarray:
        window = np.ones(n, dtype=np.float64)
        ramp = self.config.ramp_samples

        if ramp > 0:
            fade_in = np.linspace(0.0, 1.0, ramp, endpoint=False)
            fade_out = np.linspace(1.0, 0.0, ramp, endpoint=False)
            window[:ramp] *= fade_in
            window[-ramp:] *= fade_out

        return window

class PSKAudioSamplesToBits(Stage[AudioSampleBlock, BitBlock]):
    def __init__(self, config: PSKConfig):
        self.config = config
        self.carrier_phase = 0.0
        self.sample_buffer = np.empty(0, dtype=np.float64)

        self._validate_config()

    async def process(
        self,
        stream: Stream[AudioSampleBlock],
    ) -> Stream[BitBlock]:
        async for block in stream:
            samples = np.asarray(block.data, dtype=np.float64)
            samples = np.concatenate([self.sample_buffer, samples])

            n = self.config.samples_per_symbol
            num_complete_symbols = len(samples) // n
            num_used_samples = num_complete_symbols * n

            used_samples = samples[:num_used_samples]
            self.sample_buffer = samples[num_used_samples:]

            if block.is_final and len(self.sample_buffer) > 0:
                raise ValueError(
                    "Final PSK audio block does not contain a whole number "
                    "of symbols"
                )

            bits = self._demodulate_samples(used_samples)

            yield BitBlock(
                data=bits,
                is_final=block.is_final,
                metadata=block.metadata,
            )

    def _validate_config(self) -> None:
        if self.config.samples_per_symbol <= 0:
            raise ValueError("samples_per_symbol must be positive")

        if self.config.amplitude < 0:
            raise ValueError("amplitude must be non-negative")

        if self.config.order < 2:
            raise ValueError("order must be at least 2")

        if self.config.order & (self.config.order - 1) != 0:
            raise ValueError("order must be a power of two")

    def _demodulate_samples(self, samples: np.ndarray) -> np.ndarray:
        if len(samples) == 0:
            return np.empty(0, dtype=np.uint8)

        n = self.config.samples_per_symbol

        if len(samples) % n != 0:
            raise ValueError("Number of samples must be divisible by symbol size")

        symbols = samples.reshape(-1, n)

        symbol_indices = np.array(
            [self._demodulate_symbol(symbol) for symbol in symbols],
            dtype=np.uint8,
        )

        return self._symbol_indices_to_bits(symbol_indices)

    def _demodulate_symbol(self, samples: np.ndarray) -> int:
        n = self.config.samples_per_symbol

        if len(samples) != n:
            raise ValueError(f"Expected {n} samples, got {len(samples)}")

        t = np.arange(n, dtype=np.float64) / self.config.sample_rate

        carrier_phase = (
            self.carrier_phase if self.config.continuous_phase else 0.0
        )

        phase = (
            2.0 * np.pi * self.config.carrier_frequency * t
            + carrier_phase
        )

        window = self._symbol_window(n)

        i_template = np.cos(phase) * window
        q_template = np.sin(phase) * window

        i = np.dot(samples, i_template)
        q = np.dot(samples, q_template)

        if self.config.continuous_phase:
            self.carrier_phase = (
                carrier_phase
                + 2.0
                * np.pi
                * self.config.carrier_frequency
                * n
                / self.config.sample_rate
            ) % (2.0 * np.pi)

        # Because the modulator uses sin(carrier + symbol_phase):
        #
        #   sin(a + phi) = sin(a) cos(phi) + cos(a) sin(phi)
        #
        # Therefore:
        #   q ~ cos(phi)
        #   i ~ sin(phi)
        #
        # Hence atan2(i, q), not atan2(q, i).
        estimated_phase = np.arctan2(i, q)
        estimated_phase = estimated_phase % (2.0 * np.pi)

        symbol_phases = self.config.symbol_phases % (2.0 * np.pi)

        distances = np.abs(
            np.angle(np.exp(1j * (estimated_phase - symbol_phases)))
        )

        symbol_index = int(np.argmin(distances))

        return symbol_index

    def _symbol_indices_to_bits(self, symbol_indices: np.ndarray) -> np.ndarray:
        bps = self.config.bits_per_symbol

        bits = np.empty(len(symbol_indices) * bps, dtype=np.uint8)

        for i, symbol_index in enumerate(symbol_indices):
            for j in range(bps):
                shift = bps - 1 - j
                bits[i * bps + j] = (int(symbol_index) >> shift) & 1

        return bits

    def _symbol_window(self, n: int) -> np.ndarray:
        window = np.ones(n, dtype=np.float64)
        ramp = self.config.ramp_samples

        if ramp > 0:
            fade_in = np.linspace(0.0, 1.0, ramp, endpoint=False)
            fade_out = np.linspace(1.0, 0.0, ramp, endpoint=False)
            window[:ramp] *= fade_in
            window[-ramp:] *= fade_out

        return window