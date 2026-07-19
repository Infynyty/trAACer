from __future__ import annotations

import asyncio
import socket
import threading
import time
from contextlib import suppress
from dataclasses import dataclass
from typing import Callable, Literal

import matplotlib.pyplot as plt
import numpy as np
import qrcode
import uvicorn

from traacer.network_stack.layer.base import (
    AudioSampleBlock,
    BitBlock,
    ProcessorStage,
    Stream,
)
from traacer.network_stack.layer.device_layer.WebserverDeviceLayer import (
    WebserverDeviceLayerReceiverSource,
    WebserverDeviceLayerSenderSink,
)
from traacer.network_stack.layer.physical_layer.modulation.fsk import (
    BitsToFSKSymbols,
    FSKConfig,
    FSKSymbolsToAudioSamples,
    FSKSymbolsToBits,
    PacketAudioSamplesToFSKSymbols,
)
from traacer.network_stack.layer.physical_layer.modulation.psk import (
    BitsToPSKAudioSamples,
    PSKAudioSamplesToBits,
    PSKConfig,
)
from traacer.network_stack.layer.physical_layer.synchronization import (
    FindPackets,
    PacketAudioSampleBlock,
    create_chirp_preamble,
)
from traacer.receiver.server import app, cert_path, key_path


SAMPLE_RATE = 48_000
SYMBOL_RATE = 50.0
NUM_RANDOM_BITS = 24
SUPPORTED_ORDERS = (2, 4, 8, 16)

FSK_CENTER_FREQUENCY_HZ = 8_000.0
FSK_FREQUENCY_SPACING_HZ = 800.0
PSK_CARRIER_FREQUENCY_HZ = 4_000.0
AMPLITUDE = 0.8

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
    """Own a Uvicorn thread and synchronously join it during cleanup."""

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
            name="modulation-exercise-webserver",
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


@dataclass(frozen=True, slots=True)
class ModulationExperimentResult:
    scheme: Literal["FSK", "PSK"]
    order: int
    seed: int
    transmitted_bits: np.ndarray
    received_bits: np.ndarray
    bit_errors: int
    preamble_score: float

    @property
    def bit_error_rate(self) -> float:
        return self.bit_errors / len(self.transmitted_bits)


def _validate_order(order: int) -> None:
    if order not in SUPPORTED_ORDERS:
        choices = ", ".join(str(value) for value in SUPPORTED_ORDERS)
        raise ValueError(f"order must be one of {choices}")


def _random_bits(order: int, seed: int) -> np.ndarray:
    _validate_order(order)
    generator = np.random.default_rng(seed)
    return generator.integers(
        0,
        2,
        size=NUM_RANDOM_BITS,
        dtype=np.uint8,
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
    return AMPLITUDE * np.concatenate(
        (
            np.tile(upward_chirp, PREAMBLE_CHIRP_REPETITIONS),
            np.tile(downward_chirp, PREAMBLE_CHIRP_REPETITIONS),
            guard,
        )
    )


def _fsk_config(order: int) -> FSKConfig:
    _validate_order(order)
    return FSKConfig.from_spacing(
        sample_rate=SAMPLE_RATE,
        symbol_rate=SYMBOL_RATE,
        order=order,
        center_frequency=FSK_CENTER_FREQUENCY_HZ,
        frequency_spacing=FSK_FREQUENCY_SPACING_HZ,
        amplitude=AMPLITUDE,
        continuous_phase=True,
    )


def _psk_config(order: int) -> PSKConfig:
    _validate_order(order)
    return PSKConfig(
        sample_rate=SAMPLE_RATE,
        symbol_rate=SYMBOL_RATE,
        carrier_frequency=PSK_CARRIER_FREQUENCY_HZ,
        order=order,
        amplitude=AMPLITUDE,
        ramp_duration=0.002,
        continuous_phase=True,
        phase_offset=0.0,
    )


async def _single_bit_block(bits: np.ndarray) -> Stream[BitBlock]:
    yield BitBlock(data=bits, is_final=True)


async def _collect_audio(stream: Stream[AudioSampleBlock]) -> np.ndarray:
    blocks = [np.asarray(block.data, dtype=np.float64) async for block in stream]
    if not blocks:
        raise RuntimeError("The modulator produced no audio")
    return np.concatenate(blocks)


async def _modulate_fsk(bits: np.ndarray, config: FSKConfig) -> np.ndarray:
    symbols = ProcessorStage(BitsToFSKSymbols(config)).process(
        _single_bit_block(bits)
    )
    return await _collect_audio(FSKSymbolsToAudioSamples(config).process(symbols))


async def _modulate_psk(bits: np.ndarray, config: PSKConfig) -> np.ndarray:
    return await _collect_audio(
        BitsToPSKAudioSamples(config).process(_single_bit_block(bits))
    )


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
    raise RuntimeError("No complete modulation packet was received")


def _fsk_signal_space_coordinates(
    samples: np.ndarray,
    config: FSKConfig,
) -> np.ndarray:
    """Return one M-dimensional matched-filter vector per FSK symbol."""

    samples = np.asarray(samples, dtype=np.float64).reshape(-1)
    if len(samples) % config.samples_per_symbol:
        raise ValueError("FSK packet does not contain complete symbols")

    symbols = samples.reshape(-1, config.samples_per_symbol)
    n = np.arange(config.samples_per_symbol, dtype=np.float64)
    phases = (
        2.0
        * np.pi
        * config.frequencies[:, None]
        * n[None, :]
        / config.sample_rate
    )
    sin_references = np.sin(phases)
    cos_references = np.cos(phases)
    sin_energy = np.sum(np.square(sin_references), axis=1)
    cos_energy = np.sum(np.square(cos_references), axis=1)
    in_phase = symbols @ sin_references.T / sin_energy[None, :]
    quadrature = symbols @ cos_references.T / cos_energy[None, :]
    return np.hypot(in_phase, quadrature)


def _psk_signal_space_coordinates(
    samples: np.ndarray,
    config: PSKConfig,
) -> np.ndarray:
    """Return complex I/Q coordinates matching the existing PSK receiver."""

    samples = np.asarray(samples, dtype=np.float64).reshape(-1)
    if len(samples) % config.samples_per_symbol:
        raise ValueError("PSK packet does not contain complete symbols")

    symbols = samples.reshape(-1, config.samples_per_symbol)
    n = config.samples_per_symbol
    time_axis = np.arange(n, dtype=np.float64) / config.sample_rate
    window = np.ones(n, dtype=np.float64)
    if config.ramp_samples:
        ramp = config.ramp_samples
        window[:ramp] *= np.linspace(0.0, 1.0, ramp, endpoint=False)
        window[-ramp:] *= np.linspace(1.0, 0.0, ramp, endpoint=False)

    coordinates = np.empty(len(symbols), dtype=np.complex128)
    carrier_phase = 0.0
    for index, symbol in enumerate(symbols):
        phase = (
            2.0 * np.pi * config.carrier_frequency * time_axis
            + carrier_phase
        )
        cosine_reference = np.cos(phase) * window
        sine_reference = np.sin(phase) * window
        quadrature = np.dot(symbol, cosine_reference) / np.dot(
            cosine_reference,
            cosine_reference,
        )
        in_phase = np.dot(symbol, sine_reference) / np.dot(
            sine_reference,
            sine_reference,
        )
        coordinates[index] = in_phase + 1j * quadrature

        if config.continuous_phase:
            carrier_phase = (
                carrier_phase
                + 2.0
                * np.pi
                * config.carrier_frequency
                * n
                / config.sample_rate
            ) % (2.0 * np.pi)

    return coordinates


def _plot_fsk_signal_space(
    packet: PacketAudioSampleBlock,
    config: FSKConfig,
) -> None:
    coordinates = _fsk_signal_space_coordinates(packet.data, config)
    detected_symbols = np.argmax(coordinates, axis=1)

    figure, axis = plt.subplots(figsize=(11, 5), constrained_layout=True)
    image = axis.imshow(
        coordinates.T,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        vmin=0.0,
        extent=(-0.5, len(coordinates) - 0.5, -0.5, config.order - 0.5),
    )
    axis.scatter(
        np.arange(len(coordinates)),
        detected_symbols,
        marker="x",
        color="white",
        linewidths=1.2,
        label="Largest coordinate",
    )
    axis.set_xticks(np.arange(len(coordinates)))
    axis.set_yticks(
        np.arange(config.order),
        labels=[f"{frequency:g}" for frequency in config.frequencies],
    )
    axis.set_xlabel("Received symbol index")
    axis.set_ylabel("FSK basis frequency [Hz]")
    axis.set_title(
        f"Received {config.order}-FSK signal space "
        "(M-dimensional matched-filter coordinates)"
    )
    axis.legend(loc="upper right")
    figure.colorbar(image, ax=axis, label="Coordinate magnitude")
    plt.show()


def _plot_psk_signal_space(
    packet: PacketAudioSampleBlock,
    config: PSKConfig,
) -> None:
    coordinates = _psk_signal_space_coordinates(packet.data, config)
    received_level = float(np.median(np.abs(coordinates)))
    if received_level > 1e-12:
        display_coordinates = coordinates * config.amplitude / received_level
    else:
        display_coordinates = coordinates
    ideal = config.amplitude * np.exp(1j * config.symbol_phases)
    time_position = np.arange(len(coordinates))

    figure, axis = plt.subplots(figsize=(7, 7), constrained_layout=True)
    scatter = axis.scatter(
        display_coordinates.real,
        display_coordinates.imag,
        c=time_position,
        cmap="viridis",
        s=55,
        label="Received symbols",
        zorder=3,
    )
    axis.scatter(
        ideal.real,
        ideal.imag,
        marker="x",
        color="black",
        s=70,
        linewidths=1.4,
        label="Ideal symbols",
        zorder=4,
    )
    for symbol_index, point in enumerate(ideal):
        axis.annotate(
            str(symbol_index),
            (point.real, point.imag),
            xytext=(5, 5),
            textcoords="offset points",
        )

    plot_limit = 1.2 * max(
        config.amplitude,
        float(np.max(np.abs(display_coordinates), initial=0.0)),
    )
    axis.axhline(0.0, color="0.5", linewidth=0.8)
    axis.axvline(0.0, color="0.5", linewidth=0.8)
    axis.set_xlim(-plot_limit, plot_limit)
    axis.set_ylim(-plot_limit, plot_limit)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("Gain-normalized in-phase coordinate")
    axis.set_ylabel("Gain-normalized quadrature coordinate")
    axis.set_title(f"Received {config.order}-PSK signal space")
    axis.grid(True, alpha=0.25)
    axis.legend(loc="upper right")
    if len(coordinates) > 1:
        figure.colorbar(scatter, ax=axis, label="Received symbol index")
    plt.show()


async def _decode_fsk_packet(
    packet: PacketAudioSampleBlock,
    config: FSKConfig,
) -> np.ndarray:
    async def packet_stream() -> Stream[PacketAudioSampleBlock]:
        yield packet

    symbols = ProcessorStage(PacketAudioSamplesToFSKSymbols(config)).process(
        packet_stream()
    )
    bits = ProcessorStage(FSKSymbolsToBits(config)).process(symbols)
    async for block in bits:
        return np.asarray(block.data, dtype=np.uint8)
    raise RuntimeError("The FSK demodulator produced no bits")


async def _decode_psk_packet(
    packet: PacketAudioSampleBlock,
    config: PSKConfig,
) -> np.ndarray:
    async def packet_stream() -> Stream[AudioSampleBlock]:
        yield packet

    bits = PSKAudioSamplesToBits(config).process(packet_stream())
    async for block in bits:
        return np.asarray(block.data, dtype=np.uint8)
    raise RuntimeError("The PSK demodulator produced no bits")


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
    predicate: Callable[[dict[str, object]], bool],
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


def _print_result(result: ModulationExperimentResult) -> None:
    transmitted = "".join(str(int(bit)) for bit in result.transmitted_bits)
    received = "".join(str(int(bit)) for bit in result.received_bits)
    print(f"{result.scheme} order:       {result.order}")
    print(f"Random seed:     {result.seed}")
    print(f"Transmitted:     {transmitted}")
    print(f"Received:        {received}")
    print(
        f"Bit errors:      {result.bit_errors} / "
        f"{len(result.transmitted_bits)}"
    )
    print(f"Bit error rate:  {result.bit_error_rate:.3f}")
    print(f"Preamble score:  {result.preamble_score:.3f}")


async def _run_experiment(
    *,
    scheme: Literal["FSK", "PSK"],
    order: int,
    seed: int,
) -> ModulationExperimentResult:
    bits = _random_bits(order, seed)
    if scheme == "FSK":
        config = _fsk_config(order)
        payload = await _modulate_fsk(bits, config)
    else:
        config = _psk_config(order)
        payload = await _modulate_psk(bits, config)

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
            _first_packet(receiver, packet_num_samples=len(payload))
        )
        await _wait_for_status(
            sender,
            lambda status: status.get("download_active") is True,
            timeout_seconds=10.0,
            timeout_message="Timed out while connecting the audio receiver",
        )
        await asyncio.wait_for(
            sender.play(_framed_audio(payload)),
            timeout=RECEIVE_TIMEOUT_SECONDS,
        )
        packet = await asyncio.wait_for(
            packet_task,
            timeout=RECEIVE_TIMEOUT_SECONDS,
        )

        if scheme == "FSK":
            received_bits = await _decode_fsk_packet(packet, config)
            _plot_fsk_signal_space(packet, config)
        else:
            received_bits = await _decode_psk_packet(packet, config)
            _plot_psk_signal_space(packet, config)

        compared_length = min(len(bits), len(received_bits))
        bit_errors = int(
            np.count_nonzero(bits[:compared_length] != received_bits[:compared_length])
            + abs(len(bits) - len(received_bits))
        )
        result = ModulationExperimentResult(
            scheme=scheme,
            order=order,
            seed=seed,
            transmitted_bits=bits,
            received_bits=received_bits,
            bit_errors=bit_errors,
            preamble_score=packet.correlation_score,
        )
        _print_result(result)
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


async def run_fsk_experiment(
    *,
    order: int = 2,
    seed: int = 0,
) -> ModulationExperimentResult:

    return await _run_experiment(scheme="FSK", order=order, seed=seed)


async def run_psk_experiment(
    *,
    order: int = 2,
    seed: int = 0,
) -> ModulationExperimentResult:

    return await _run_experiment(scheme="PSK", order=order, seed=seed)


if __name__ == "__main__":
    asyncio.run(run_fsk_experiment())
