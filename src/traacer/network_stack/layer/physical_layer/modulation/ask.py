from dataclasses import dataclass

import numpy as np

from traacer.metrics.plot import plot_signal, plot_fft, plot_signal_with_signal_boundaries, q
from traacer.network_stack.layer.base import BitBlock, AudioSampleBlock, Stage, Stream
from traacer.network_stack.layer.device_layer.DeviceLayer import DeviceLayerSender, DeviceLayerReceiver
from traacer.network_stack.layer.payload_layer.LinkLayer import LinkLayerReceiver
from traacer.network_stack.layer.physical_layer.PhysicalLayerUtil import bytes_to_bits, manchester_encode, \
    manchester_decode, bits_to_bytes
from traacer.network_stack.layer.physical_layer.Synchronization import create_zadoff_chu_preamble, find_preamble_start
from traacer.network_stack.packet import LinkLayerPacket, PhysicalLayerPacket, DeviceLayerPacket

import matplotlib.pyplot as plt


class OOKPhysicalLayerSender:
    def __init__(
        self,
        device_layer_sender: DeviceLayerSender,
        amplitude_reduction_factor: float,
        sample_rate: int = 44100,
        carrier_freq: float = 2000.0,
        bit_duration: float = 0.2,
        ramp_duration: float = 0.002,
    ):
        self.sample_rate = sample_rate
        self.carrier_freq = carrier_freq
        self.bit_duration = bit_duration
        self.ramp_duration = ramp_duration

        self.amplitude = 1.0 / amplitude_reduction_factor
        self.samples_per_bit = int(self.sample_rate * self.bit_duration)
        self.ramp_samples = min(
            int(self.sample_rate * self.ramp_duration),
            self.samples_per_bit // 2,
        )

        if self.ramp_samples > 0:
            self.ramp = (
                0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, self.ramp_samples))
            ).astype(np.float32)
        else:
            self.ramp = np.empty(0, dtype=np.float32)

        self.device_layer_sender = device_layer_sender

    def send_down(self, packet: LinkLayerPacket):
        bits = np.fromiter(
            bytes_to_bits(packet.data.tobytes()),
            dtype=np.uint8,
        )

        if bits.size == 0:
            self.device_layer_sender.send_down(
                PhysicalLayerPacket(np.empty(0, dtype=np.float32))
            )
            return

        encoded_bits = manchester_encode(bits)
        envelope = np.repeat(encoded_bits.astype(np.float32), self.samples_per_bit)

        if self.ramp_samples > 0:
            padded = np.pad(envelope, (1, 1))
            rising = np.flatnonzero((padded[1:-1] == 1.0) & (padded[:-2] == 0.0))
            falling = np.flatnonzero((padded[1:-1] == 1.0) & (padded[2:] == 0.0))

            for start in rising:
                stop = min(start + self.ramp_samples, envelope.size)
                envelope[start:stop] *= self.ramp[: stop - start]

            for end in falling:
                start = max(end - self.ramp_samples + 1, 0)
                n = end - start + 1
                envelope[start:end + 1] *= self.ramp[-n:][::-1]

        total_samples = envelope.size
        t = np.arange(total_samples, dtype=np.float64) / self.sample_rate
        carrier = np.sin(2.0 * np.pi * self.carrier_freq * t)

        signal = (self.amplitude * envelope * carrier).astype(np.float32)

        plot_signal_with_signal_boundaries(signal, symbol_length=int(self.sample_rate*self.bit_duration), xlim=50000)
        q.put(signal)

        signal = np.concat([create_zadoff_chu_preamble(), signal])

        self.device_layer_sender.send_down(PhysicalLayerPacket(signal))


class OOKPhysicalLayerReceiver:
    def __init__(
        self,
        link_layer_receiver: LinkLayerReceiver,
        sample_rate: int = 44100,
        carrier_freq: float = 2000.0,
        bit_duration: float = 0.2,
        decision_threshold: float | None = None,
    ):
        self.sample_rate = sample_rate
        self.carrier_freq = carrier_freq
        self.bit_duration = bit_duration
        self.samples_per_bit = int(self.sample_rate * self.bit_duration)

        t = np.arange(self.samples_per_bit) / self.sample_rate
        self.carrier = np.sin(2 * np.pi * self.carrier_freq * t).astype(np.float32)

        self.link_layer_receiver = link_layer_receiver
        self.decision_threshold = decision_threshold




    def _detect_bits(self, signal: np.ndarray) -> list[int]:
        if signal.size == 0:
            return []

        envelope = np.abs(signal).astype(np.float32)

        smooth_window = max(1, self.samples_per_bit // 8)
        if smooth_window > 1:
            kernel = np.ones(smooth_window, dtype=np.float32) / smooth_window
            envelope = np.convolve(envelope, kernel, mode="same")

        n_bits = len(envelope) // self.samples_per_bit
        if n_bits == 0:
            return []

        metrics = np.empty(n_bits, dtype=np.float32)

        for i in range(n_bits):
            s = i * self.samples_per_bit
            e = s + self.samples_per_bit
            metrics[i] = np.mean(envelope[s:e])

        if self.decision_threshold is not None:
            threshold = self.decision_threshold
            return [1 if metric > threshold else 0 for metric in metrics]

        win = 4
        padded = np.pad(metrics, (win, win), mode="edge")

        bits = []
        for i in range(n_bits):
            local = padded[i:i + 2 * win]
            local_min = np.min(local)
            local_max = np.max(local)
            threshold = 0.5 * (local_min + local_max)
            bits.append(1 if metrics[i] > threshold else 0)

        return bits

    def send_up(self, packet: DeviceLayerPacket):
        signal = np.asarray(packet.data, dtype=np.float32)
        preamble = create_zadoff_chu_preamble()
        signal_start_index = find_preamble_start(signal, preamble) + len(preamble)
        signal = signal[signal_start_index:]

        detected_bits = self._detect_bits(signal)
        decoded_bits = manchester_decode(detected_bits)
        data = bits_to_bytes(decoded_bits)

        plot_signal_with_signal_boundaries(signal, self.samples_per_bit, 50000)
        plot_fft(signal)
        print(f"Detected bits (with Manchester Encoding): {detected_bits}")
        print(f"Decoded bits: {decoded_bits}")
        q.put(signal)
        self.link_layer_receiver.send_up(
            PhysicalLayerPacket(np.frombuffer(data, dtype=np.uint8))
        )

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


from __future__ import annotations

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
                f"ASK symbol index must be in [0, {self.config.order - 1}], "
                f"got {symbol_index}"
            )

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

        amplitude = self.config.amplitude_levels[symbol_index]
        symbol = amplitude * carrier

        window = self._symbol_window(n)
        symbol *= window

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

class ASKAudioSamplesToBits(Stage[AudioSampleBlock, BitBlock]):
    def __init__(self, config: ASKConfig):
        self.config = config
        self.phase = 0.0
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
                    "Final ASK audio block does not contain a whole number "
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

        window = self._symbol_window(n)
        template = carrier * window

        template_energy = np.dot(template, template)

        if template_energy <= 0:
            raise ValueError("ASK matched-filter template has zero energy")

        estimated_amplitude = np.dot(samples, template) / template_energy

        levels = self.config.amplitude_levels
        symbol_index = int(np.argmin(np.abs(levels - estimated_amplitude)))

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