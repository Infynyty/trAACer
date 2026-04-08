import numpy as np

from traacer.network_stack.layer.device_layer.DeviceLayer import DeviceLayerSender, DeviceLayerReceiver
from traacer.network_stack.layer.payload_layer.LinkLayer import LinkLayerReceiver
from traacer.network_stack.packet import LinkLayerPacket, PhysicalLayerPacket, DeviceLayerPacket


class OOKPhysicalLayerSender:

    def __init__(
        self,
        device_layer_sender: DeviceLayerSender,
        amplitude_reduction_factor: float,
        sample_rate: int = 16000,
        carrier_freq: float = 2000.0,
        bit_duration: float = 0.01,
    ):
        self.sample_rate = sample_rate
        self.carrier_freq = carrier_freq
        self.bit_duration = bit_duration

        self.amplitude = 1.0 / amplitude_reduction_factor

        self.samples_per_bit = int(self.sample_rate * self.bit_duration)

        t = np.arange(self.samples_per_bit) / self.sample_rate
        self.carrier = np.sin(2 * np.pi * self.carrier_freq * t).astype(np.float32)

        self.zero = np.zeros(self.samples_per_bit, dtype=np.float32)

        self.device_layer_sender = device_layer_sender

    def _bytes_to_bits(self, data: bytes):
        for byte in data:
            for i in range(8):
                yield (byte >> (7 - i)) & 1

    def send_down(self, packet: LinkLayerPacket):
        symbols = []

        for bit in self._bytes_to_bits(packet.data.tobytes()):
            if bit == 1:
                symbols.append(self.amplitude * self.carrier)
            else:
                symbols.append(self.zero)

        signal = np.concatenate(symbols).astype(np.float32)

        self.device_layer_sender.send_down(PhysicalLayerPacket(signal))

class OOKPhysicalLayerReceiver:

    def __init__(
        self,
        link_layer_receiver: LinkLayerReceiver,
        sample_rate: int = 16000,
        carrier_freq: float = 2000.0,
        bit_duration: float = 0.01,
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

    def _bits_to_bytes(self, bits: list[int]) -> bytes:
        n_bytes = len(bits) // 8
        out = bytearray()

        for i in range(n_bytes):
            byte = 0
            for bit in bits[i * 8:(i + 1) * 8]:
                byte = (byte << 1) | bit
            out.append(byte)

        return bytes(out)

    def _detect_bits(self, signal: np.ndarray) -> list[int]:
        n_symbols = len(signal) // self.samples_per_bit
        if n_symbols == 0:
            return []

        signal = signal[:n_symbols * self.samples_per_bit]
        symbols = signal.reshape(n_symbols, self.samples_per_bit)

        scores = symbols @ self.carrier

        if self.decision_threshold is None:
            threshold = 0.5 * np.max(scores) if len(scores) else 0.0
        else:
            threshold = self.decision_threshold

        return [1 if score > threshold else 0 for score in scores]

    def send_up(self, packet: DeviceLayerPacket):
        signal = np.asarray(packet.data, dtype=np.float32)

        bits = self._detect_bits(signal)
        data = self._bits_to_bytes(bits)

        self.link_layer_receiver.send_up(PhysicalLayerPacket(np.frombuffer(data, dtype=np.uint8)))

