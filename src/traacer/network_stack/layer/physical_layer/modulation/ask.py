from dataclasses import dataclass


from traacer.network_stack.layer.base import BitBlock, AudioSampleBlock, Stage, Stream
import numpy as np

from traacer.network_stack.layer.physical_layer.synchronization import PacketAudioSampleBlock


@dataclass(frozen=True)
class OOKConfig:
    sample_rate: int
    symbol_rate: float
    carrier_frequency: float
    amplitude: float = 0.8
    ramp_duration: float = 0.002
    continuous_phase: bool = True

    @property
    def samples_per_symbol(self) -> int:
        return int(round(self.sample_rate / self.symbol_rate))

    @property
    def ramp_samples(self) -> int:
        return min(
            int(round(self.ramp_duration * self.sample_rate)),
            self.samples_per_symbol // 2,
        )


class BitsToOOKAudioSamples(Stage[BitBlock, AudioSampleBlock]):
    def __init__(self, config: OOKConfig):
        self.config = config
        self.phase = 0.0

        if config.samples_per_symbol <= 0:
            raise ValueError("samples_per_symbol must be positive")

        if config.amplitude < 0:
            raise ValueError("amplitude must be non-negative")

    async def process(
        self,
        stream: Stream[BitBlock],
    ) -> Stream[AudioSampleBlock]:
        async for block in stream:
            samples = self._modulate_bits(block.data)

            yield AudioSampleBlock(
                data=samples,
                is_final=block.is_final,
                metadata=block.metadata,
            )

    def _modulate_bits(self, bits: np.ndarray) -> np.ndarray:
        output = np.empty(
            len(bits) * self.config.samples_per_symbol,
            dtype=np.float64,
        )

        offset = 0

        for bit in bits:
            symbol = self._modulate_symbol(int(bit))
            output[offset:offset + self.config.samples_per_symbol] = symbol
            offset += self.config.samples_per_symbol

        return output

    def _modulate_symbol(self, bit: int) -> np.ndarray:
        n = self.config.samples_per_symbol
        t = np.arange(n, dtype=np.float64) / self.config.sample_rate

        phase = self.phase if self.config.continuous_phase else 0.0
        carrier = np.sin(
            2.0 * np.pi * self.config.carrier_frequency * t + phase
        )

        if self.config.continuous_phase:
            self.phase = (
                phase
                + 2.0 * np.pi * self.config.carrier_frequency * n / self.config.sample_rate
            ) % (2.0 * np.pi)

        if bit == 0:
            return np.zeros(n, dtype=np.float64)

        if bit != 1:
            raise ValueError(f"OOK bits must be 0 or 1, got {bit}")

        symbol = self.config.amplitude * carrier
        ramp = self.config.ramp_samples

        if ramp > 0:
            fade_in = np.linspace(0.0, 1.0, ramp, endpoint=False)
            fade_out = np.linspace(1.0, 0.0, ramp, endpoint=False)
            symbol[:ramp] *= fade_in
            symbol[-ramp:] *= fade_out

        return symbol


from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class ASKConfig:
    sample_rate: int
    symbol_rate: float
    carrier_frequency: float
    order: int = 4
    amplitude: float = 0.8
    ramp_duration: float = 0.002
    continuous_phase: bool = False

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
    def amplitude_levels(self) -> np.ndarray:
        """
        Natural M-ASK levels from 0 to amplitude.

        For order=4 and amplitude=0.8:
            [0.0, 0.266..., 0.533..., 0.8]
        """
        return np.linspace(
            0.0,
            self.amplitude,
            self.order,
            dtype=np.float64,
        )

class BitsToASKAudioSamples(Stage[BitBlock, AudioSampleBlock]):
    def __init__(self, config: ASKConfig):
        self.config = config
        self.phase = 0.0
        self.bit_buffer = np.empty(0, dtype=np.uint8)
        self.pilot_pending = True

        self._validate_config()

    async def process(
        self,
        stream: Stream[BitBlock],
    ) -> Stream[AudioSampleBlock]:
        async for block in stream:
            bits = np.asarray(block.data, dtype=np.uint8)

            if np.any((bits != 0) & (bits != 1)):
                raise ValueError("ASK input bits must only contain 0 and 1")

            bits = np.concatenate([self.bit_buffer, bits])

            num_complete_symbols = len(bits) // self.config.bits_per_symbol
            num_used_bits = num_complete_symbols * self.config.bits_per_symbol

            used_bits = bits[:num_used_bits]
            self.bit_buffer = bits[num_used_bits:]

            if block.is_final and len(self.bit_buffer) > 0:
                raise ValueError(
                    "Final ASK bit block does not contain a whole number "
                    f"of {self.config.bits_per_symbol}-bit symbols"
                )

            output_parts = []

            if self.pilot_pending:
                output_parts.append(self._modulate_amplitude(1.0))
                self.pilot_pending = False

            payload_samples = self._modulate_bits(used_bits)

            if len(payload_samples) > 0:
                output_parts.append(payload_samples)

            if output_parts:
                samples = np.concatenate(output_parts)
            else:
                samples = np.empty(0, dtype=np.float64)

            yield AudioSampleBlock(
                data=samples,
                is_final=block.is_final,
                metadata=block.metadata,
            )

            if block.is_final:
                self.pilot_pending = True

    def _validate_config(self) -> None:
        if self.config.samples_per_symbol <= 0:
            raise ValueError("samples_per_symbol must be positive")

        if self.config.amplitude < 0:
            raise ValueError("amplitude must be non-negative")

        if self.config.order < 2:
            raise ValueError("order must be at least 2")

        if self.config.order & (self.config.order - 1) != 0:
            raise ValueError("order must be a power of two")

        if self.config.bits_per_symbol <= 0:
            raise ValueError("bits_per_symbol must be positive")

    def _modulate_bits(self, bits: np.ndarray) -> np.ndarray:
        if len(bits) == 0:
            return np.empty(0, dtype=np.float64)

        symbols = self._bits_to_symbol_indices(bits)

        output = np.empty(
            len(symbols) * self.config.samples_per_symbol,
            dtype=np.float64,
        )

        offset = 0

        for symbol_index in symbols:
            amplitude = self.config.amplitude_levels[int(symbol_index)]
            symbol = self._modulate_amplitude(amplitude)

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

    def _modulate_amplitude(self, amplitude: float) -> np.ndarray:
        n = self.config.samples_per_symbol
        t = np.arange(n, dtype=np.float64) / self.config.sample_rate

        phase = self.phase if self.config.continuous_phase else 0.0

        carrier = np.sin(
            2.0 * np.pi * self.config.carrier_frequency * t + phase
        )

        if self.config.continuous_phase:
            self.phase = (
                phase
                + 2.0
                * np.pi
                * self.config.carrier_frequency
                * n
                / self.config.sample_rate
            ) % (2.0 * np.pi)

        return amplitude * carrier * self._symbol_window(n)

    def _symbol_window(self, n: int) -> np.ndarray:
        window = np.ones(n, dtype=np.float64)
        ramp = self.config.ramp_samples

        if ramp > 0:
            fade_in = np.linspace(0.0, 1.0, ramp, endpoint=False)
            fade_out = np.linspace(1.0, 0.0, ramp, endpoint=False)
            window[:ramp] *= fade_in
            window[-ramp:] *= fade_out

        return window


class ASKAudioSamplesToBits(Stage[PacketAudioSampleBlock, BitBlock]):
    def __init__(self, config: ASKConfig):
        self.config = config
        self.phase = 0.0
        self.sample_buffer = np.empty(0, dtype=np.float64)
        self.pilot_pending = True
        self.channel_gain: float | None = None

        self._validate_config()

    async def process(
        self,
        stream: Stream[PacketAudioSampleBlock],
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
                    "Final ASK audio block does not contain a whole number "
                    "of symbols"
                )

            bits = self._demodulate_samples(used_samples)

            yield BitBlock(
                data=bits,
                is_final=block.is_final,
                metadata=block.metadata,
            )

            if block.is_final:
                self.pilot_pending = True
                self.channel_gain = None

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
            raise ValueError(
                "Number of samples must be divisible by symbol size"
            )

        symbols = samples.reshape(-1, n)
        payload_start = 0

        if self.pilot_pending:
            self.channel_gain = self._estimate_amplitude(symbols[0])

            if self.channel_gain <= 0:
                raise ValueError(
                    "ASK pilot produced a non-positive channel-gain estimate"
                )

            self.pilot_pending = False
            payload_start = 1

        payload_symbols = symbols[payload_start:]

        if len(payload_symbols) == 0:
            return np.empty(0, dtype=np.uint8)

        symbol_indices = np.array(
            [
                self._demodulate_symbol(symbol)
                for symbol in payload_symbols
            ],
            dtype=np.uint8,
        )

        return self._symbol_indices_to_bits(symbol_indices)

    def _demodulate_symbol(self, samples: np.ndarray) -> int:
        if self.channel_gain is None:
            raise RuntimeError(
                "ASK channel gain has not been estimated from the pilot"
            )

        estimated_amplitude = self._estimate_amplitude(samples)
        estimated_levels = (
            self.channel_gain
            * np.asarray(self.config.amplitude_levels, dtype=np.float64)
        )

        return int(
            np.argmin(
                np.abs(estimated_levels - estimated_amplitude)
            )
        )

    def _estimate_amplitude(self, samples: np.ndarray) -> float:
        n = self.config.samples_per_symbol

        if len(samples) != n:
            raise ValueError(f"Expected {n} samples, got {len(samples)}")

        t = np.arange(n, dtype=np.float64) / self.config.sample_rate
        angular_phase = (
                2.0
                * np.pi
                * self.config.carrier_frequency
                * t
        )

        window = self._symbol_window(n)

        sin_template = np.sin(angular_phase) * window
        cos_template = np.cos(angular_phase) * window

        sin_energy = np.dot(sin_template, sin_template)
        cos_energy = np.dot(cos_template, cos_template)

        if sin_energy <= 0 or cos_energy <= 0:
            raise ValueError("ASK correlation template has zero energy")

        in_phase = np.dot(samples, sin_template) / sin_energy
        quadrature = np.dot(samples, cos_template) / cos_energy

        return float(np.hypot(in_phase, quadrature))

    def _symbol_indices_to_bits(
        self,
        symbol_indices: np.ndarray,
    ) -> np.ndarray:
        bps = self.config.bits_per_symbol

        bits = np.empty(
            len(symbol_indices) * bps,
            dtype=np.uint8,
        )

        for i, symbol_index in enumerate(symbol_indices):
            for j in range(bps):
                shift = bps - 1 - j
                bits[i * bps + j] = (
                    int(symbol_index) >> shift
                ) & 1

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
