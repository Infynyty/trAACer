import numpy as np

from traacer.metrics.plot import plot_signal, plot_fft, q
from traacer.network_stack.layer.device_layer.DeviceLayer import DeviceLayerSender
from traacer.network_stack.layer.payload_layer.LinkLayer import LinkLayerReceiver
from traacer.network_stack.layer.physical_layer.PhysicalLayerUtil import manchester_decode, bits_to_bytes, \
    bytes_to_bits, manchester_encode
from traacer.network_stack.packet import LinkLayerPacket, PhysicalLayerPacket, DeviceLayerPacket

import matplotlib.pyplot as plt

class FSKPhysicalLayerSender:
    def __init__(
        self,
        device_layer_sender: DeviceLayerSender,
        amplitude_reduction_factor: float,
        sample_rate: int = 44100,
        freq_zero: float = 1200.0,
        freq_one: float = 2200.0,
        bit_duration: float = 0.05,
        ramp_duration: float = 0.002,
        use_manchester: bool = True,
    ):
        self.sample_rate = sample_rate
        self.freq_zero = freq_zero
        self.freq_one = freq_one
        self.bit_duration = bit_duration
        self.ramp_duration = ramp_duration
        self.use_manchester = use_manchester

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

        t = np.arange(self.samples_per_bit, dtype=np.float64) / self.sample_rate
        self.symbol_zero = np.sin(2.0 * np.pi * self.freq_zero * t).astype(np.float32)
        self.symbol_one = np.sin(2.0 * np.pi * self.freq_one * t).astype(np.float32)

        self.device_layer_sender = device_layer_sender


    def _apply_ramp(self, symbol: np.ndarray) -> np.ndarray:
        if self.ramp_samples == 0:
            return symbol.copy()

        out = symbol.copy()
        out[:self.ramp_samples] *= self.ramp
        out[-self.ramp_samples:] *= self.ramp[::-1]
        return out

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

        if self.use_manchester:
            bits = manchester_encode(bits)

        zero_symbol = self._apply_ramp(self.amplitude * self.symbol_zero)
        one_symbol = self._apply_ramp(self.amplitude * self.symbol_one)

        symbols = [one_symbol if bit else zero_symbol for bit in bits]
        signal = np.concatenate(symbols).astype(np.float32)

        q.put(signal)
        self.device_layer_sender.send_down(PhysicalLayerPacket(signal))


class FSKPhysicalLayerReceiver:
    def __init__(
        self,
        link_layer_receiver: LinkLayerReceiver,
        sample_rate: int = 44100,
        freq_zero: float = 1200.0,
        freq_one: float = 2200.0,
        bit_duration: float = 0.05,
        use_manchester: bool = True,
    ):
        self.sample_rate = sample_rate
        self.freq_zero = freq_zero
        self.freq_one = freq_one
        self.bit_duration = bit_duration
        self.use_manchester = use_manchester
        self.samples_per_bit = int(self.sample_rate * self.bit_duration)

        t = np.arange(self.samples_per_bit, dtype=np.float64) / self.sample_rate

        ref_zero_i = np.cos(2.0 * np.pi * self.freq_zero * t)
        ref_zero_q = np.sin(2.0 * np.pi * self.freq_zero * t)
        ref_one_i = np.cos(2.0 * np.pi * self.freq_one * t)
        ref_one_q = np.sin(2.0 * np.pi * self.freq_one * t)

        self.ref_zero_i = ref_zero_i.astype(np.float32)
        self.ref_zero_q = ref_zero_q.astype(np.float32)
        self.ref_one_i = ref_one_i.astype(np.float32)
        self.ref_one_q = ref_one_q.astype(np.float32)

        self.link_layer_receiver = link_layer_receiver


    def _tone_energy(self, symbol: np.ndarray, ref_i: np.ndarray, ref_q: np.ndarray) -> float:
        i = np.dot(symbol, ref_i)
        q = np.dot(symbol, ref_q)
        return float(i * i + q * q)

    def _detect_bits(self, signal: np.ndarray) -> list[int]:
        n_bits = len(signal) // self.samples_per_bit
        if n_bits == 0:
            return []

        bits = []

        for i in range(n_bits):
            s = i * self.samples_per_bit
            e = s + self.samples_per_bit
            symbol = signal[s:e]

            energy_zero = self._tone_energy(symbol, self.ref_zero_i, self.ref_zero_q)
            energy_one = self._tone_energy(symbol, self.ref_one_i, self.ref_one_q)

            bits.append(1 if energy_one > energy_zero else 0)

        return bits

    def send_up(self, packet: DeviceLayerPacket):
        signal = np.asarray(packet.data, dtype=np.float32)

        detected_bits = self._detect_bits(signal)

        if self.use_manchester:
            detected_bits = manchester_decode(detected_bits)

        data = bits_to_bytes(detected_bits)

        plot_signal(signal)
        plot_fft(signal)
        print(f"Decoded bits: {detected_bits}")
        q.put(signal)
        self.link_layer_receiver.send_up(
            PhysicalLayerPacket(np.frombuffer(data, dtype=np.uint8))
        )