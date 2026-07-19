from __future__ import annotations

import asyncio
import socket
import threading
import time
from contextlib import suppress
from dataclasses import dataclass
from typing import Literal

import numpy as np
import qrcode
import uvicorn

from traacer.network_stack.layer.base import AudioSampleBlock, BitBlock, Stream
from traacer.network_stack.layer.device_layer.WebserverDeviceLayer import (
    WebserverDeviceLayerReceiverSource,
    WebserverDeviceLayerSenderSink,
)
from traacer.network_stack.layer.physical_layer.modulation.ask import (
    ASKAudioSamplesToBits,
    ASKConfig,
    BitsToASKAudioSamples,
)
from traacer.network_stack.layer.physical_layer.synchronization import (
    FindPackets,
    PacketAudioSampleBlock,
    create_chirp_preamble,
)
from traacer.receiver.server import app, cert_path, key_path


CodingScheme = Literal["No coding", "Repeat-3", "Hamming (7,4)"]
CODING_SCHEMES: tuple[CodingScheme, ...] = (
    "No coding",
    "Repeat-3",
    "Hamming (7,4)",
)

SAMPLE_RATE = 48_000
CARRIER_FREQUENCY_HZ = 4_000.0
SYMBOL_RATE = 50.0
ASK_ORDER = 8
ASK_AMPLITUDE = 0.8

PAYLOAD_SEED = 2026
PAYLOAD_LENGTH = 48
FIXED_PAYLOAD = np.random.default_rng(PAYLOAD_SEED).integers(
    0,
    2,
    size=PAYLOAD_LENGTH,
    dtype=np.uint8,
)

PREAMBLE_START_FREQUENCY_HZ = 500.0
PREAMBLE_END_FREQUENCY_HZ = 16_000.0
PREAMBLE_CHIRP_DURATION_SECONDS = 0.02
PREAMBLE_CHIRP_REPETITIONS = 2
PREAMBLE_GUARD_DURATION_SECONDS = 0.20
PREAMBLE_DETECTION_THRESHOLD = 0.001
LEADING_SILENCE_SECONDS = 0.5
TRAILING_SILENCE_SECONDS = 0.5
RECEIVE_TIMEOUT_SECONDS = 30.0
SERVER_PORT = 8_000
SERVER_START_TIMEOUT_SECONDS = 10.0
SERVER_STOP_TIMEOUT_SECONDS = 5.0


@dataclass(frozen=True, slots=True)
class ChannelCodingResult:
    coding: CodingScheme
    transmitted_payload: np.ndarray
    transmitted_coded_bits: np.ndarray
    received_coded_bits: np.ndarray
    decoded_payload: np.ndarray
    coded_bit_errors: int
    payload_bit_errors: int
    preamble_score: float

    @property
    def payload_bit_error_rate(self) -> float:
        return self.payload_bit_errors / len(self.transmitted_payload)

    @property
    def number_of_ask_symbols(self) -> int:
        return len(self.transmitted_coded_bits) // 3


def fixed_payload() -> np.ndarray:
    """Return the fixed random payload used by every exercise run."""

    return FIXED_PAYLOAD.copy()


def encode_payload(bits: np.ndarray, coding: CodingScheme) -> np.ndarray:
    data = _validate_bits(bits)
    _validate_coding(coding)

    if coding == "No coding":
        encoded = data.copy()
    elif coding == "Repeat-3":
        encoded = np.repeat(data, 3)
    else:
        if len(data) % 4:
            raise ValueError("Hamming (7,4) requires a multiple of four bits")
        words = data.reshape(-1, 4)
        d1, d2, d3, d4 = words.T
        encoded = np.column_stack(
            (
                d1 ^ d2 ^ d4,
                d1 ^ d3 ^ d4,
                d1,
                d2 ^ d3 ^ d4,
                d2,
                d3,
                d4,
            )
        ).reshape(-1)

    if len(encoded) % 3:
        raise ValueError("The coded payload must contain complete 8-ASK symbols")
    return encoded.astype(np.uint8, copy=False)


def decode_payload(bits: np.ndarray, coding: CodingScheme) -> np.ndarray:
    data = _validate_bits(bits)
    _validate_coding(coding)

    if coding == "No coding":
        return data.copy()

    if coding == "Repeat-3":
        if len(data) % 3:
            raise ValueError("Repeat-3 received an incomplete codeword")
        groups = data.reshape(-1, 3)
        return (groups.sum(axis=1) >= 2).astype(np.uint8)

    if len(data) % 7:
        raise ValueError("Hamming (7,4) received an incomplete codeword")
    words = data.reshape(-1, 7).copy()
    s1 = words[:, 0] ^ words[:, 2] ^ words[:, 4] ^ words[:, 6]
    s2 = words[:, 1] ^ words[:, 2] ^ words[:, 5] ^ words[:, 6]
    s4 = words[:, 3] ^ words[:, 4] ^ words[:, 5] ^ words[:, 6]
    error_positions = s1 + 2 * s2 + 4 * s4
    error_rows = np.flatnonzero(error_positions)
    if len(error_rows):
        words[error_rows, error_positions[error_rows] - 1] ^= 1
    return words[:, [2, 4, 5, 6]].reshape(-1)


def _validate_bits(bits: np.ndarray) -> np.ndarray:
    data = np.asarray(bits, dtype=np.uint8).reshape(-1)
    if not len(data):
        raise ValueError("The bit sequence cannot be empty")
    if np.any((data != 0) & (data != 1)):
        raise ValueError("The bit sequence may only contain 0 and 1")
    return data


def _validate_coding(coding: str) -> None:
    if coding not in CODING_SCHEMES:
        choices = ", ".join(CODING_SCHEMES)
        raise ValueError(f"coding must be one of: {choices}")


def _ask_config() -> ASKConfig:
    return ASKConfig(
        sample_rate=SAMPLE_RATE,
        symbol_rate=SYMBOL_RATE,
        carrier_frequency=CARRIER_FREQUENCY_HZ,
        order=ASK_ORDER,
        amplitude=ASK_AMPLITUDE,
        ramp_duration=0.002,
        continuous_phase=True,
    )


def _create_preamble() -> np.ndarray:
    upward_chirp = create_chirp_preamble(
        start_frequency=PREAMBLE_START_FREQUENCY_HZ,
        end_frequency=PREAMBLE_END_FREQUENCY_HZ,
        duration_in_sec=PREAMBLE_CHIRP_DURATION_SECONDS,
        sampling_frequency=SAMPLE_RATE,
    )
    downward_chirp = create_chirp_preamble(
        start_frequency=PREAMBLE_END_FREQUENCY_HZ,
        end_frequency=PREAMBLE_START_FREQUENCY_HZ,
        duration_in_sec=PREAMBLE_CHIRP_DURATION_SECONDS,
        sampling_frequency=SAMPLE_RATE,
    )
    guard = np.zeros(
        round(PREAMBLE_GUARD_DURATION_SECONDS * SAMPLE_RATE),
        dtype=np.float64,
    )
    return ASK_AMPLITUDE * np.concatenate(
        (
            np.tile(upward_chirp, PREAMBLE_CHIRP_REPETITIONS),
            np.tile(downward_chirp, PREAMBLE_CHIRP_REPETITIONS),
            guard,
        )
    )


async def _single_bit_block(bits: np.ndarray) -> Stream[BitBlock]:
    yield BitBlock(data=bits, is_final=True)


async def _modulate(bits: np.ndarray, config: ASKConfig) -> np.ndarray:
    blocks = [
        np.asarray(block.data, dtype=np.float64)
        async for block in BitsToASKAudioSamples(config).process(
            _single_bit_block(bits)
        )
    ]
    if not blocks:
        raise RuntimeError("The 8-ASK modulator produced no audio")
    return np.concatenate(blocks)


async def _framed_audio(payload: np.ndarray) -> Stream[AudioSampleBlock]:
    leading_silence = np.zeros(
        round(LEADING_SILENCE_SECONDS * SAMPLE_RATE),
        dtype=np.float64,
    )
    trailing_silence = np.zeros(
        round(TRAILING_SILENCE_SECONDS * SAMPLE_RATE),
        dtype=np.float64,
    )
    yield AudioSampleBlock(
        data=np.concatenate(
            (leading_silence, _create_preamble(), payload, trailing_silence)
        ),
        is_final=True,
        metadata={"sample_rate": SAMPLE_RATE},
    )


async def _first_packet(
    receiver: WebserverDeviceLayerReceiverSource,
    packet_num_samples: int,
) -> PacketAudioSampleBlock:
    packets = FindPackets(
        preamble=_create_preamble(),
        packet_num_samples=packet_num_samples,
        threshold=PREAMBLE_DETECTION_THRESHOLD,
    ).process(receiver.stream())
    async for packet in packets:
        return packet
    raise RuntimeError("No complete 8-ASK packet was received")


async def _demodulate(
    packet: PacketAudioSampleBlock,
    config: ASKConfig,
) -> np.ndarray:
    async def packet_stream() -> Stream[PacketAudioSampleBlock]:
        yield packet

    decoded_stream = ASKAudioSamplesToBits(config).process(packet_stream())
    async for block in decoded_stream:
        return np.asarray(block.data, dtype=np.uint8).reshape(-1)
    raise RuntimeError("The 8-ASK demodulator produced no bits")


def _count_bit_errors(expected: np.ndarray, actual: np.ndarray) -> int:
    comparable = min(len(expected), len(actual))
    return int(
        np.count_nonzero(expected[:comparable] != actual[:comparable])
        + abs(len(expected) - len(actual))
    )


def _format_bits(bits: np.ndarray) -> str:
    return "".join(str(int(bit)) for bit in bits)


def _format_error_markers(expected: np.ndarray, actual: np.ndarray) -> str:
    comparable = min(len(expected), len(actual))
    markers = [
        "^" if expected[index] != actual[index] else " "
        for index in range(comparable)
    ]
    markers.extend("^" for _ in range(abs(len(expected) - len(actual))))
    return "".join(markers)


def print_result(result: ChannelCodingResult) -> None:
    print("\n8-ASK channel-coding result")
    print(f"  Coding:                    {result.coding}")
    print(f"  Fixed payload seed:        {PAYLOAD_SEED}")
    print(f"  Coded bits sent:           {len(result.transmitted_coded_bits)}")
    print(f"  8-ASK payload symbols:     {result.number_of_ask_symbols}")
    print(
        f"  Coded-bit errors:          {result.coded_bit_errors} / "
        f"{len(result.transmitted_coded_bits)}"
    )
    print(
        f"  Bit errors after decoding: {result.payload_bit_errors} / "
        f"{len(result.transmitted_payload)}"
    )
    print(f"  Payload BER:               {result.payload_bit_error_rate:.3f}")
    print(f"  Preamble score:            {result.preamble_score:.3f}")
    print(f"\n  Sent:    {_format_bits(result.transmitted_payload)}")
    print(f"  Decoded: {_format_bits(result.decoded_payload)}")
    print(
        "           "
        + _format_error_markers(
            result.transmitted_payload,
            result.decoded_payload,
        )
        + "  (^ = bit error)"
    )


def _get_lan_ip() -> str:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80))
        return sock.getsockname()[0]
    finally:
        sock.close()


def _print_receiver_url(port: int) -> None:
    url = f"https://{_get_lan_ip()}:{port}"
    qr = qrcode.QRCode(border=1)
    qr.add_data(url)
    qr.make(fit=True)
    print(f"\nReceiver URL: {url}\n")
    for row in qr.get_matrix():
        print("".join("██" if cell else "  " for cell in row))
    print()


class _ManagedWebserver:
    def __init__(self) -> None:
        self.port = SERVER_PORT
        self.server = uvicorn.Server(
            uvicorn.Config(
                app,
                host="0.0.0.0",
                port=self.port,
                reload=False,
                ssl_certfile=str(cert_path),
                ssl_keyfile=str(key_path),
                log_level="critical",
                timeout_graceful_shutdown=1,
            )
        )
        self.error: BaseException | None = None
        self.thread = threading.Thread(
            target=self._run,
            daemon=False,
            name="channel-coding-exercise-webserver",
        )

    def _run(self) -> None:
        try:
            self.server.run()
        except BaseException as error:
            self.error = error

    async def start(self) -> None:
        _print_receiver_url(self.port)
        self.thread.start()
        deadline = time.monotonic() + SERVER_START_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            if self.server.started:
                return
            if not self.thread.is_alive():
                if self.error is not None:
                    raise RuntimeError("The receiver webserver failed") from self.error
                raise RuntimeError(
                    f"The receiver webserver could not bind to port {self.port}"
                )
            await asyncio.sleep(0.05)
        raise TimeoutError("Timed out while starting the receiver webserver")

    def stop(self) -> None:
        if not self.thread.is_alive():
            return
        self.server.should_exit = True
        self.thread.join(timeout=SERVER_STOP_TIMEOUT_SECONDS)
        if self.thread.is_alive():
            self.server.force_exit = True
            self.thread.join(timeout=SERVER_STOP_TIMEOUT_SECONDS)
        if self.thread.is_alive():
            raise RuntimeError("The receiver webserver thread did not stop")


async def _start_waiting_for_microphone(
    sender: WebserverDeviceLayerSenderSink,
    timeout_seconds: float = 10.0,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if sender._try_start_waiting() is not None:
            return
        await asyncio.sleep(0.1)
    raise TimeoutError("Timed out while preparing the browser microphone session")


async def _wait_for_status(
    sender: WebserverDeviceLayerSenderSink,
    predicate,
    *,
    timeout_seconds: float,
    timeout_message: str,
) -> dict[str, object]:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            status = sender._get_status()
        except Exception:
            await asyncio.sleep(0.1)
            continue
        if predicate(status):
            return status
        await asyncio.sleep(0.1)
    raise TimeoutError(timeout_message)


async def run_channel_coding_experiment(
    *,
    coding: CodingScheme = "No coding",
) -> ChannelCodingResult:
    """Send the fixed payload through the real browser audio channel using 8-ASK."""

    _validate_coding(coding)
    payload_bits = fixed_payload()
    coded_bits = encode_payload(payload_bits, coding)
    config = _ask_config()
    modulated_audio = await _modulate(coded_bits, config)

    print(f"Fixed payload ({PAYLOAD_LENGTH} bits, seed {PAYLOAD_SEED}):")
    print(_format_bits(payload_bits))
    print(f"Selected coding: {coding}")

    webserver = _ManagedWebserver()
    sender: WebserverDeviceLayerSenderSink | None = None
    packet_task: asyncio.Task[PacketAudioSampleBlock] | None = None

    try:
        await webserver.start()
        sender = WebserverDeviceLayerSenderSink(
            server_url=f"https://127.0.0.1:{SERVER_PORT}",
            sample_rate=SAMPLE_RATE,
            cert_path=cert_path,
        )
        receiver = WebserverDeviceLayerReceiverSource(
            server_url=f"https://127.0.0.1:{SERVER_PORT}",
            target_sample_rate=SAMPLE_RATE,
            cert_path=cert_path,
        )

        await _start_waiting_for_microphone(sender)
        print("Open the receiver URL shown above and start the microphone.")
        await _wait_for_status(
            sender,
            lambda status: bool(
                status.get("device_ready") and status.get("sample_rate")
            ),
            timeout_seconds=310.0,
            timeout_message="Timed out waiting for the browser microphone",
        )

        packet_task = asyncio.create_task(
            _first_packet(receiver, packet_num_samples=len(modulated_audio))
        )
        await _wait_for_status(
            sender,
            lambda status: status.get("download_active") is True,
            timeout_seconds=10.0,
            timeout_message="Timed out while connecting the audio receiver",
        )
        await asyncio.wait_for(
            sender.play(_framed_audio(modulated_audio)),
            timeout=RECEIVE_TIMEOUT_SECONDS,
        )
        packet = await asyncio.wait_for(
            packet_task,
            timeout=RECEIVE_TIMEOUT_SECONDS,
        )

        received_coded_bits = await _demodulate(packet, config)
        decoded_payload = decode_payload(received_coded_bits, coding)
        result = ChannelCodingResult(
            coding=coding,
            transmitted_payload=payload_bits,
            transmitted_coded_bits=coded_bits,
            received_coded_bits=received_coded_bits,
            decoded_payload=decoded_payload,
            coded_bit_errors=_count_bit_errors(coded_bits, received_coded_bits),
            payload_bit_errors=_count_bit_errors(payload_bits, decoded_payload),
            preamble_score=packet.correlation_score,
        )
        print_result(result)
        return result
    finally:
        if packet_task is not None:
            if not packet_task.done():
                packet_task.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await packet_task
        if sender is not None:
            with suppress(Exception):
                sender._stop_blocking()
            sender.session.close()
        webserver.stop()


if __name__ == "__main__":
    asyncio.run(run_channel_coding_experiment())
