import traceback

import asyncio
import socket
import threading

import numpy as np
import qrcode
import uvicorn

from traacer.network_stack.layer.base import Stream, CharBlock, CharToBytes, BytesToBits, ProcessorStage, BitsToBytes, \
    BytesToChar, user_input_source, PrintCharSink, run_webserver, string_source
from traacer.network_stack.layer.device_layer.WebserverDeviceLayer import WebserverDeviceLayerSenderSink, \
    WebserverDeviceLayerReceiverSource
from traacer.network_stack.layer.physical_layer.error_correction import Repeat3, AddGuardSilence, Repeat3Corrector, \
    Hamming74Encode, Hamming74Decode
from traacer.network_stack.layer.physical_layer.framing import AddBitLengthHeader, RemoveBitLengthHeader
from traacer.network_stack.layer.physical_layer.modulation.ask import ASKConfig, BitsToASKAudioSamples, \
    ASKAudioSamplesToBits
from traacer.network_stack.layer.physical_layer.modulation.iq import IQConfig, IQSymbolsToAudioSamples, \
    AudioSamplesToIQSymbols
from traacer.network_stack.layer.physical_layer.modulation.ofdm import OFDMConfig, BitsToQAMSymbols, \
    QAMSymbolsToOFDMFrame, IQSymbolsToQAMSymbols, QAMSymbolsToBits
from traacer.network_stack.layer.physical_layer.synchronization import PrependPreamble, FindPackets, \
    create_chirp_preamble
from traacer.receiver.server import cert_path

async def main() -> None:
    source = string_source("hi")
    webserver_thread = threading.Thread(
        target=run_webserver,
        daemon=False,
    )
    webserver_thread.start()


    ask_config = ASKConfig(48000, 100, 4000)

    preamble = np.concat(
        [np.tile(create_chirp_preamble(start_frequency=500, end_frequency=16000, duration_in_sec=0.02), 2),
         np.tile(create_chirp_preamble(start_frequency=16000, end_frequency=500, duration_in_sec=0.02), 2),
         np.zeros(10000, dtype=np.float64)
         ])

    char_to_bytes = CharToBytes()
    bytes_to_bits = BytesToBits()
    bits_with_header = AddBitLengthHeader(header_bits=8)
    bits_with_hamming = Hamming74Encode()
    ask_symbols = BitsToASKAudioSamples(ask_config)
    symbols_with_preamble = PrependPreamble(preamble)

    byte_stream = char_to_bytes.process(source)
    bit_stream = bytes_to_bits.process(byte_stream)
    bit_stream_with_header = bits_with_header.process(bit_stream)
    bit_stream_with_hamming = bits_with_hamming.process(bit_stream_with_header)
    ask_symbol_stream = ask_symbols.process(bit_stream_with_hamming)
    audio_with_preamble_stream = symbols_with_preamble.process(ask_symbol_stream)

    sender = WebserverDeviceLayerSenderSink(
        server_url="https://127.0.0.1:8000",
        sample_rate=ask_config.sample_rate,
        cert_path=cert_path
    )

    receiver = WebserverDeviceLayerReceiverSource(
        cert_path=cert_path
    )

    find_packets = FindPackets(
        preamble=preamble,
        packet_num_samples=48000*5,
        threshold=0.6,
    )

    packet_audio_to_ask_symbols = ASKAudioSamplesToBits(ask_config)
    bits_without_hamming = ProcessorStage(Hamming74Decode())
    bits_without_header = ProcessorStage(RemoveBitLengthHeader(8))
    bits_to_bytes = ProcessorStage(BitsToBytes())
    bytes_to_char = BytesToChar()
    sink = PrintCharSink()

    received_audio_stream = receiver.stream()
    detected_packet_stream = find_packets.process(received_audio_stream)
    rx_bits = packet_audio_to_ask_symbols.process(detected_packet_stream)
    rx_bits_without_hamming = bits_without_hamming.process(rx_bits)
    rx_bit_length_stream = bits_without_header.process(rx_bits_without_hamming)
    rx_byte_stream = bits_to_bytes.process(rx_bit_length_stream)
    rx_char_stream = bytes_to_char.process(rx_byte_stream)

    async def run_tx() -> None:
        try:
            await sender.consume(audio_with_preamble_stream)
        except Exception as e:
            traceback.print_exc()
            print("Got exception")
            await run_tx()

    async def run_rx() -> None:
        await sink.consume(rx_char_stream)

    await asyncio.gather(run_rx(), run_tx())

asyncio.run(main())