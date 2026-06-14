import asyncio
import socket
import threading

import qrcode
import uvicorn

from traacer.network_stack.layer.base import CharToBytes, BytesToBits, ProcessorStage, BitsToBytes, BytesToChar, \
    user_input_source
from traacer.network_stack.layer.device_layer.WebserverDeviceLayer import WebserverDeviceLayerSenderSink, \
    WebserverDeviceLayerReceiverSource
from traacer.network_stack.layer.physical_layer.ErrorCorrection import Repeat3Corrector, Repeat3
from traacer.network_stack.layer.physical_layer.FSKPhysicalLayer import FSKConfig, BitsToFSKSymbols, \
    FSKSymbolsToAudioSamples, PacketAudioSamplesToFSKSymbols, FSKSymbolsToBits
from traacer.network_stack.layer.physical_layer.Length import RemoveBitLengthHeader, AddBitLengthHeader
from traacer.network_stack.layer.physical_layer.Synchronization import create_chirp_preamble, PrependPreamble, \
    FindPackets
from traacer.receiver.server import app, cert_path, key_path


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

    config = FSKConfig.from_spacing(
        sample_rate=48000,
        symbol_rate=50,
        order=2,
        center_frequency=5000,
        frequency_spacing=500,
        amplitude=0.8,
        continuous_phase=True,
    )

    preamble = create_chirp_preamble(duration_in_sec=0.2)

    char_to_bytes = CharToBytes()
    bytes_to_bits = BytesToBits()
    bits_with_header = AddBitLengthHeader(header_bits=4)
    bits_to_duplicate = Repeat3()
    bits_to_fsk_symbols = ProcessorStage(BitsToFSKSymbols(config))
    fsk_symbols_to_audio = FSKSymbolsToAudioSamples(config)
    symbols_with_preamble = PrependPreamble(preamble)

    byte_stream = char_to_bytes.process(source)
    bit_stream = bytes_to_bits.process(byte_stream)
    bit_stream_with_header = bits_with_header.process(bit_stream)
    repeated_bit_stream = bits_to_duplicate.process(bit_stream_with_header)
    symbol_stream = bits_to_fsk_symbols.process(repeated_bit_stream)
    audio_stream = fsk_symbols_to_audio.process(symbol_stream)
    audio_with_preamble_stream = symbols_with_preamble.process(audio_stream)

    sender = WebserverDeviceLayerSenderSink(
        server_url="https://127.0.0.1:8000",
        sample_rate=config.sample_rate,
        cert_path="/Users/kasimirstadie/PycharmProjects/BT/trAACer/src/traacer/receiver/certs/dev-cert.pem"
    )

    receiver = WebserverDeviceLayerReceiverSource(
        cert_path="/Users/kasimirstadie/PycharmProjects/BT/trAACer/src/traacer/receiver/certs/dev-cert.pem"
    )

    payload_bits = 5 * 8
    header_bits = 4
    repeated_bits = 3 * (header_bits + payload_bits)

    find_packets = FindPackets(
        preamble=preamble,
        packet_num_samples=(repeated_bits // config.bits_per_symbol) * config.samples_per_symbol,
        threshold=0.06,
    )

    packet_audio_to_fsk_symbols = ProcessorStage(
        PacketAudioSamplesToFSKSymbols(config)
    )
    fsk_symbols_to_bits = ProcessorStage(FSKSymbolsToBits(config))
    bits_to_deduped = ProcessorStage(Repeat3Corrector())
    bits_without_header = ProcessorStage(RemoveBitLengthHeader(4))
    bits_to_bytes = ProcessorStage(BitsToBytes())
    bytes_to_char = BytesToChar()

    received_audio_stream = receiver.stream()
    detected_packet_stream = find_packets.process(received_audio_stream)
    rx_symbol_stream = packet_audio_to_fsk_symbols.process(detected_packet_stream)
    rx_bit_stream = fsk_symbols_to_bits.process(rx_symbol_stream)
    rx_deduped_bit_stream = bits_to_deduped.process(rx_bit_stream)
    rx_bit_length_stream = bits_without_header.process(rx_deduped_bit_stream)
    rx_byte_stream = bits_to_bytes.process(rx_bit_length_stream)
    rx_char_stream = bytes_to_char.process(rx_byte_stream)

    async def run_tx() -> None:
        await sender.consume(audio_with_preamble_stream)

    async def run_rx() -> None:
        async for block in rx_char_stream:
            print(block.data)

    await asyncio.gather(run_rx(), run_tx())


asyncio.run(main())