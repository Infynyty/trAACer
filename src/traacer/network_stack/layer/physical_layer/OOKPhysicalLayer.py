import numpy as np

from traacer.metrics.plot import plot_signal, plot_fft, plot_signal_with_signal_boundaries, q
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