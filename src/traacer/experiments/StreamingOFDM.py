import asyncio
import socket
import threading
from random import sample

import numpy as np
import qrcode
import uvicorn

from traacer.network_stack.layer.base import Stream, CharBlock, CharToBytes, BytesToBits, ProcessorStage, BitsToBytes, \
    BytesToChar, user_input_source
from traacer.network_stack.layer.device_layer.WebserverDeviceLayer import WebserverDeviceLayerSenderSink, \
    WebserverDeviceLayerReceiverSource
from traacer.network_stack.layer.physical_layer.ErrorCorrection import Repeat3, Repeat3Corrector, AddGuardSilence
from traacer.network_stack.layer.physical_layer.FSKPhysicalLayer import BitsToFSKSymbols, FSKSymbolsToAudioSamples, \
    PacketAudioSamplesToFSKSymbols, FSKSymbolsToBits, FSKConfig
from traacer.network_stack.layer.physical_layer.IQModulator import IQSymbolsToAudioSamples, IQConfig, \
    AudioSamplesToIQSymbols
from traacer.network_stack.layer.physical_layer.Length import AddBitLengthHeader, RemoveBitLengthHeader
from traacer.network_stack.layer.physical_layer.OFDMModulator import BitsToQAMSymbols, QAMSymbolsToOFDMFrame, \
    OFDMConfig, QAMSymbolsToBits, IQSymbolsToQAMSymbols
from traacer.network_stack.layer.physical_layer.Synchronization import create_chirp_preamble, PrependPreamble, \
    FindPackets
from traacer.receiver.server import app, cert_path, key_path


async def string_source(messages: list[str]) -> Stream[CharBlock]:
    for message in messages:
        print("Message: " + message)
        yield CharBlock(data=message, is_final=True)




def run_webserver() -> None:
    try:
        def get_lan_ip() -> str:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            try:
                sock.connect(("8.8.8.8", 80))
                return sock.getsockname()[0]
            finally:
                sock.close()

        def print_qr_to_console(url: str) -> None:
            qr = qrcode.QRCode(border=1)
            qr.add_data(url)
            qr.make(fit=True)

            print()
            print(f"Receiver URL: {url}")
            print()

            for row in qr.get_matrix():
                print("".join("██" if cell else "  " for cell in row))

            print()

        host = "0.0.0.0"
        port = 8000
        scheme = "https"

        url = f"{scheme}://{get_lan_ip()}:{port}"
        print_qr_to_console(url)

        config = uvicorn.Config(
            app,
            host=host,
            port=port,
            reload=False,
            ssl_certfile=str(cert_path),
            ssl_keyfile=str(key_path),
            log_level="critical",
        )

        server = uvicorn.Server(config)
        server.run()

    except BaseException as exc:
        print("WEBSERVER THREAD CRASHED:", repr(exc))
        raise

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
        pilot_spacing=3,
        pilot_value=1,
        num_positive_subcarriers=16
    )

    iq_config = IQConfig(
        carrier_frequency=12000,
        sample_rate=48000,
        lowpass_cutoff=1000,
        numtaps=257,
        carrier_phase_origin=0
    )

    preamble = np.concat(
        [np.tile(create_chirp_preamble(start_frequency=500, end_frequency=16000, duration_in_sec=0.02), 4),
         np.tile(create_chirp_preamble(start_frequency=16000, end_frequency=500, duration_in_sec=0.02), 4)])

    char_to_bytes = CharToBytes()
    bytes_to_bits = BytesToBits()
    bits_with_header = AddBitLengthHeader(header_bits=4)
    bits_to_duplicate = Repeat3()
    bits_to_qam = ProcessorStage(BitsToQAMSymbols(2))
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
        cert_path="/Users/kasimirstadie/PycharmProjects/BT/trAACer/src/traacer/receiver/certs/dev-cert.pem"
    )

    receiver = WebserverDeviceLayerReceiverSource(
        cert_path="/Users/kasimirstadie/PycharmProjects/BT/trAACer/src/traacer/receiver/certs/dev-cert.pem"
    )

    find_packets = FindPackets(
        preamble=preamble,
        packet_num_samples=ofdm_config.get_samples_per_ofdm_frame(),
        threshold=0.7,
    )

    packet_audio_to_iq_symbols = AudioSamplesToIQSymbols(iq_config)
    iq_symbols_to_qam_symbols = ProcessorStage(IQSymbolsToQAMSymbols(ofdm_config))
    qam_symbols_to_bits = QAMSymbolsToBits(2)
    bits_to_deduped = ProcessorStage(Repeat3Corrector())
    bits_without_header = ProcessorStage(RemoveBitLengthHeader(4))
    bits_to_bytes = ProcessorStage(BitsToBytes())
    bytes_to_char = BytesToChar()

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
        except Exception:
            print("Got exception")
            await run_tx()

    async def run_rx() -> None:
        async for block in rx_char_stream:
            print(block.data)

    await asyncio.gather(run_rx(), run_tx())


asyncio.run(main())