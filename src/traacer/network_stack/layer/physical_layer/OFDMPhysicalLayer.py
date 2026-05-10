import numpy as np

from traacer.metrics.bit_comparator import bit_queue
from traacer.metrics.plot import plot_signal
from traacer.network_stack.layer.device_layer.DeviceLayer import DeviceLayerSender
from traacer.network_stack.layer.physical_layer.IQModulator import iq_demodulate
from traacer.network_stack.layer.physical_layer.OFDMModulator import ofdm_modulate, ofdm_demodulate
from traacer.network_stack.layer.physical_layer.PhysicalLayerUtil import bytes_to_bits
from traacer.network_stack.layer.physical_layer.Synchronization import create_chirp_preamble, find_preamble_start
from traacer.network_stack.packet import LinkLayerPacket, DeviceLayerPacket, PhysicalLayerPacket


class OFDMPhysicalLayerSender:
    def __init__(self, device_layer_sender: DeviceLayerSender):
        self.device_layer_sender = device_layer_sender

    def send_down(self, packet: LinkLayerPacket):

        signal = packet.data

        audio_samples = ofdm_modulate(signal, 4, 12000, 44100, 256, 1024, 4, 1, 32)
        signal = np.concat([
            np.tile(create_chirp_preamble(start_frequency=500, end_frequency=16000, duration_in_sec=0.02), 4),
            np.tile(create_chirp_preamble(start_frequency=16000, end_frequency=500, duration_in_sec=0.02), 4),
            np.zeros(10000),
            audio_samples,
            np.zeros(10000)
        ])

        self.device_layer_sender.send_down(PhysicalLayerPacket(signal))
        return signal

class OFDMPhysicalLayerReceiver:
    def send_up(self, packet: DeviceLayerPacket):
        signal = packet.data
        preamble = np.concat(
            [np.tile(create_chirp_preamble(start_frequency=500, end_frequency=16000, duration_in_sec=0.02), 4),
             np.tile(create_chirp_preamble(start_frequency=16000, end_frequency=500, duration_in_sec=0.02), 4)])
        signal_start_index = find_preamble_start(signal, preamble) + len(preamble) + 10000

        baseband_full = iq_demodulate(
            signal,
            carrier_frequency=12000,
            sample_rate=44100,
            lowpass_cutoff=7000,
            numtaps=257,
            carrier_phase_origin=signal_start_index,
        )

        baseband_symbol = baseband_full[signal_start_index: signal_start_index + 1024 + 256]

        bits = ofdm_demodulate(
            baseband_symbol,
            n_fft=1024,
            qam_mu=4,
            cp_len=256,
            pilot_spacing=4,
            pilot_value=1,
            num_positive_subcarriers=32,
        )
        bit_queue.put(bits)
