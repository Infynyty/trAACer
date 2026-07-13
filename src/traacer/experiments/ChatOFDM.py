import traceback

import asyncio
import socket
import threading

import numpy as np
import qrcode
import uvicorn

from traacer.network_stack.layer.base import Stream, CharBlock, CharToBytes, BytesToBits, ProcessorStage, BitsToBytes, \
    BytesToChar, user_input_source, PrintCharSink, run_webserver
from traacer.network_stack.layer.device_layer.WebserverDeviceLayer import WebserverDeviceLayerSenderSink, \
    WebserverDeviceLayerReceiverSource
from traacer.network_stack.layer.physical_layer.error_correction import Repeat3, AddGuardSilence, Repeat3Corrector
from traacer.network_stack.layer.physical_layer.framing import AddBitLengthHeader, RemoveBitLengthHeader
from traacer.network_stack.layer.physical_layer.modulation.iq import IQConfig, IQSymbolsToAudioSamples, \
    AudioSamplesToIQSymbols
from traacer.network_stack.layer.physical_layer.modulation.ofdm import OFDMConfig, BitsToQAMSymbols, \
    QAMSymbolsToOFDMFrame, IQSymbolsToQAMSymbols, QAMSymbolsToBits
from traacer.network_stack.layer.physical_layer.synchronization import PrependPreamble, FindPackets, \
    create_chirp_preamble
from traacer.receiver.server import cert_path


async def main() -> None:
    source = user_input_source()
    webserver_thread = threading.Thread(
        target=run_webserver,
        daemon=False,
    )
    webserver_thread.start()


    ofdm_config = OFDMConfig(
        center_frequency=12000,
        sample_rate=48000,
        cp_len=256,
        n_fft=1024,
        pilot_spacing=8,
        pilot_value=1,
        num_positive_subcarriers=32
    )

    iq_config = IQConfig(
        carrier_frequency=12000,
        sample_rate=48000,
        lowpass_cutoff=4000,
        numtaps=257,
        carrier_phase_origin=0
    )

    preamble = np.concat(
        [np.tile(create_chirp_preamble(start_frequency=500, end_frequency=16000, duration_in_sec=0.02), 1),
         np.tile(create_chirp_preamble(start_frequency=16000, end_frequency=500, duration_in_sec=0.02), 1),
         np.zeros(10000, dtype=np.float64)
         ])

    char_to_bytes = CharToBytes()
    bytes_to_bits = BytesToBits()
    bits_with_header = AddBitLengthHeader(header_bits=8)
    bits_to_duplicate = Repeat3()
    bits_to_qam = ProcessorStage(BitsToQAMSymbols(4))
    qam_to_ofdm = QAMSymbolsToOFDMFrame(ofdm_config)
    ofdm_to_audio = IQSymbolsToAudioSamples(iq_config)
    symbols_with_preamble = PrependPreamble(preamble)
    symbols_with_guard = AddGuardSilence(10000)

    byte_stream = char_to_bytes.process(source)
    bit_stream = bytes_to_bits.process(byte_stream)
    bit_stream_with_header = bits_with_header.process(bit_stream)
    repeated_bit_stream = bits_to_duplicate.process(bit_stream_with_header)
    qam_stream = bits_to_qam.process(repeated_bit_stream)
    ofdm_frame_stream = qam_to_ofdm.process(qam_stream)
    audio_stream = ofdm_to_audio.process(ofdm_frame_stream)
    audio_with_preamble_stream = symbols_with_preamble.process(audio_stream)
    audio_with_guard_stream = symbols_with_guard.process(audio_with_preamble_stream)

    sender = WebserverDeviceLayerSenderSink(
        server_url="https://127.0.0.1:8000",
        sample_rate=ofdm_config.sample_rate,
        cert_path=cert_path
    )

    receiver = WebserverDeviceLayerReceiverSource(
        cert_path=cert_path
    )

    find_packets = FindPackets(
        preamble=preamble,
        packet_num_samples=ofdm_config.get_samples_per_ofdm_frame(),
        threshold=0.6,
    )

    packet_audio_to_iq_symbols = AudioSamplesToIQSymbols(iq_config)
    iq_symbols_to_qam_symbols = ProcessorStage(IQSymbolsToQAMSymbols(ofdm_config))
    qam_symbols_to_bits = QAMSymbolsToBits(4)
    bits_to_deduped = ProcessorStage(Repeat3Corrector())
    bits_without_header = ProcessorStage(RemoveBitLengthHeader(8))
    bits_to_bytes = ProcessorStage(BitsToBytes())
    bytes_to_char = BytesToChar()
    sink = PrintCharSink()

    received_audio_stream = receiver.stream()
    detected_packet_stream = find_packets.process(received_audio_stream)
    rx_iq_symbol_stream = packet_audio_to_iq_symbols.process(detected_packet_stream)
    rx_qam_symbol_stream = iq_symbols_to_qam_symbols.process(rx_iq_symbol_stream)
    rx_bit_stream = qam_symbols_to_bits.process(rx_qam_symbol_stream)
    rx_deduped_bit_stream = bits_to_deduped.process(rx_bit_stream)
    rx_bit_length_stream = bits_without_header.process(rx_deduped_bit_stream)
    rx_byte_stream = bits_to_bytes.process(rx_bit_length_stream)
    rx_char_stream = bytes_to_char.process(rx_byte_stream)

    async def run_tx() -> None:
        try:
            await sender.consume(audio_with_guard_stream)
        except Exception as e:
            traceback.print_exc()
            print("Got exception")
            await run_tx()

    async def run_rx() -> None:
        await sink.consume(rx_char_stream)

    await asyncio.gather(run_rx(), run_tx())

