from typing import Protocol

import numpy as np

from traacer.metrics.plot import plot_spectrogram
from traacer.network_stack.layer.device_layer.DeviceLayer import DeviceLayerSender
from traacer.network_stack.layer.physical_layer.CSSModulator import css_modulate, css_demodulate
from traacer.network_stack.layer.physical_layer.PhysicalLayerUtil import bytes_to_bits
from traacer.network_stack.layer.physical_layer.Synchronization import create_chirp_preamble, find_preamble_start, \
    create_zadoff_chu_preamble
from traacer.network_stack.packet import LinkLayerPacket, DeviceLayerPacket, PhysicalLayerPacket


class CSSPhysicalLayerSender:
    def __init__(self, device_layer_sender: DeviceLayerSender):
        self.device_layer_sender = device_layer_sender

    def send_down(self, packet: LinkLayerPacket):
        signal = np.fromiter(
            bytes_to_bits(packet.data.tobytes()),
            dtype=np.uint8,
        )

        signal = css_modulate(signal, 44100, 500, 10000, 0.1, 2)

        signal =np.concat([np.tile(create_chirp_preamble(start_frequency=500, end_frequency=16000, duration_in_sec=0.02), 4), np.tile(create_chirp_preamble(start_frequency=16000, end_frequency=500, duration_in_sec=0.02), 4), np.zeros(10000), signal])

        self.device_layer_sender.send_down(PhysicalLayerPacket(signal))

class CSSPhysicalLayerReceiver:
    def send_up(self, packet: DeviceLayerPacket):
        signal = packet.data
        plot_spectrogram(signal, 44100)
        preamble = np.concat([np.tile(create_chirp_preamble(start_frequency=500, end_frequency=16000, duration_in_sec=0.02), 4), np.tile(create_chirp_preamble(start_frequency=16000, end_frequency=500, duration_in_sec=0.02), 4)])
        signal_start_index = find_preamble_start(signal, preamble) + len(preamble) + 10000
        signal = signal[signal_start_index:]

        data = css_demodulate(signal, 44100, 500, 10000, 0.1, 2)
        print(data)
