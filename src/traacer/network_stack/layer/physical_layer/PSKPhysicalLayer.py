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
