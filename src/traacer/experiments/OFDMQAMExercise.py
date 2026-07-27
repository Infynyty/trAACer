from __future__ import annotations

import asyncio
import math
import socket
import threading
import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import qrcode
import requests
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
from traacer.network_stack.layer.physical_layer.modulation.iq import (
    IQConfig,
    IQSymbolBlock,
    IQSymbolsToAudioSamples,
    iq_demodulate,
)
from traacer.network_stack.layer.physical_layer.modulation.ofdm import (
    BitsToQAMSymbols,
    IQSymbolsToQAMSymbols,
    OFDMConfig,
    QAMSymbolsToOFDMFrame,
    get_used_indices,
    qam_gray_lookup_table,
    qam_to_bits,
)
from traacer.network_stack.layer.physical_layer.synchronization import (
    FindPackets,
    PacketAudioSampleBlock,
    create_chirp_preamble,
)
from traacer.receiver.server import app, cert_path, key_path


SUPPORTED_QAM_BITS_PER_SYMBOL = (2, 4, 6)
PAYLOAD_LENGTH_BITS = 2_688
PAYLOAD_SEED = 0x0F_D0

OFDM_PEAK_AMPLITUDE = 0.7
IQ_FILTER_TAPS = 257

PREAMBLE_START_FREQUENCY_HZ = 500.0
PREAMBLE_END_FREQUENCY_HZ = 16_000.0
PREAMBLE_DURATION_SECONDS = 0.08
PREAMBLE_AMPLITUDE = 0.8
PREAMBLE_DETECTION_THRESHOLD = 0.15
LEADING_SILENCE_SECONDS = 0.5
GUARD_DURATION_SECONDS = 0.08
TRAILING_SILENCE_SECONDS = 0.4

SERVER_PORT = 8_000
SERVER_START_TIMEOUT_SECONDS = 10.0
SERVER_STOP_TIMEOUT_SECONDS = 5.0
MICROPHONE_TIMEOUT_SECONDS = 310.0
RECEIVE_TIMEOUT_SECONDS = 45.0


@dataclass(frozen=True, slots=True)
class QAMConfig:
    """QAM settings that students can change between exercise runs."""

    bits_per_symbol: int = 4

    @property
    def order(self) -> int:
        return 2**self.bits_per_symbol


@dataclass(frozen=True, slots=True)
class OFDMQAMExerciseResult:
    qam_config: QAMConfig
    ofdm_config: OFDMConfig
    transmitted_bits: np.ndarray
    received_bits: np.ndarray
    equalized_symbols: np.ndarray
    bit_errors: int
    rms_evm: float
    timing_offset_samples: int
    preamble_score: float
    number_of_frames: int
    data_subcarriers_per_frame: int
    pilot_subcarriers_per_frame: int

    @property
    def bit_error_rate(self) -> float:
        return self.bit_errors / len(self.transmitted_bits)


def fixed_payload(seed: int = PAYLOAD_SEED) -> np.ndarray:
    """Return the reproducible payload used to compare configurations."""

    return np.random.default_rng(seed).integers(
        0,
        2,
        size=PAYLOAD_LENGTH_BITS,
        dtype=np.uint8,
    )


def _subcarrier_layout(
    config: OFDMConfig,
) -> tuple[np.ndarray, np.ndarray]:
    used_indices = get_used_indices(
        config.n_fft,
        config.num_positive_subcarriers,
    )
    pilot_indices = used_indices[:: config.pilot_spacing]
    data_indices = np.setdiff1d(used_indices, pilot_indices)
    return pilot_indices, data_indices


def _occupied_baseband_bandwidth(config: OFDMConfig) -> float:
    return (
        config.num_positive_subcarriers
        * config.sample_rate
        / config.n_fft
    )


def _validate_configuration(
    qam_config: QAMConfig,
    ofdm_config: OFDMConfig,
) -> None:
    if qam_config.bits_per_symbol not in SUPPORTED_QAM_BITS_PER_SYMBOL:
        choices = ", ".join(
            str(value) for value in SUPPORTED_QAM_BITS_PER_SYMBOL
        )
        raise ValueError(
            f"QAM bits_per_symbol must be one of {choices}"
        )
    if ofdm_config.sample_rate <= 0:
        raise ValueError("OFDM sample_rate must be positive")
    if ofdm_config.n_fft < 64 or ofdm_config.n_fft % 2:
        raise ValueError("OFDM n_fft must be an even integer of at least 64")
    if not 0 <= ofdm_config.cp_len <= ofdm_config.n_fft:
        raise ValueError("OFDM cp_len must be between 0 and n_fft")
    if ofdm_config.pilot_spacing <= 0:
        raise ValueError("OFDM pilot_spacing must be positive")
    if ofdm_config.pilot_value == 0:
        raise ValueError("OFDM pilot_value must be non-zero")
    if not (
        0
        < ofdm_config.num_positive_subcarriers
        < ofdm_config.n_fft // 2
    ):
        raise ValueError(
            "OFDM num_positive_subcarriers must be between 1 and "
            "n_fft / 2"
        )

    nyquist_frequency = ofdm_config.sample_rate / 2
    if not 0 < ofdm_config.center_frequency < nyquist_frequency:
        raise ValueError(
            "OFDM center_frequency must lie between 0 Hz and Nyquist"
        )

    occupied_bandwidth = _occupied_baseband_bandwidth(ofdm_config)
    if (
        ofdm_config.center_frequency - occupied_bandwidth <= 0
        or ofdm_config.center_frequency + occupied_bandwidth
        >= nyquist_frequency
    ):
        raise ValueError(
            "The active OFDM subcarriers do not fit between 0 Hz and "
            "Nyquist. Increase n_fft, reduce num_positive_subcarriers, "
            "or move center_frequency."
        )

    pilot_indices, data_indices = _subcarrier_layout(ofdm_config)
    positive_pilots = pilot_indices < ofdm_config.n_fft // 2
    negative_pilots = pilot_indices > ofdm_config.n_fft // 2
    if not np.any(positive_pilots) or not np.any(negative_pilots):
        raise ValueError(
            "pilot_spacing leaves one half of the OFDM spectrum without "
            "a pilot. Reduce pilot_spacing or use more subcarriers."
        )
    if not len(data_indices):
        raise ValueError("The selected OFDM layout has no data subcarriers")


def _make_iq_config(config: OFDMConfig) -> IQConfig:
    occupied_bandwidth = _occupied_baseband_bandwidth(config)
    nyquist_frequency = config.sample_rate / 2
    passband_margin = min(
        config.center_frequency,
        nyquist_frequency - config.center_frequency,
    )
    lowpass_cutoff = min(
        0.9 * passband_margin,
        max(2_500.0, 1.5 * occupied_bandwidth),
    )
    if lowpass_cutoff <= occupied_bandwidth:
        raise ValueError(
            "The OFDM bandwidth leaves no transition band for the IQ "
            "low-pass filter"
        )
    return IQConfig(
        carrier_frequency=config.center_frequency,
        sample_rate=config.sample_rate,
        lowpass_cutoff=lowpass_cutoff,
        numtaps=IQ_FILTER_TAPS,
        carrier_phase_origin=0,
    )


def _create_preamble(sample_rate: int) -> np.ndarray:
    if PREAMBLE_END_FREQUENCY_HZ >= sample_rate / 2:
        raise ValueError(
            "The exercise preamble must remain below Nyquist"
        )
    return PREAMBLE_AMPLITUDE * create_chirp_preamble(
        start_frequency=PREAMBLE_START_FREQUENCY_HZ,
        end_frequency=PREAMBLE_END_FREQUENCY_HZ,
        duration_in_sec=PREAMBLE_DURATION_SECONDS,
        sampling_frequency=sample_rate,
    )


async def _frame_bit_source(
    bits: np.ndarray,
    bits_per_frame: int,
    metadata: dict[str, object],
) -> Stream[BitBlock]:
    for start in range(0, len(bits), bits_per_frame):
        stop = start + bits_per_frame
        yield BitBlock(
            data=bits[start:stop],
            is_final=stop >= len(bits),
            metadata=metadata,
        )


async def _single_iq_block(
    samples: np.ndarray,
    metadata: dict[str, object],
) -> Stream[IQSymbolBlock]:
    yield IQSymbolBlock(
        data=samples,
        is_final=True,
        metadata=metadata,
    )


async def _modulate_payload(
    payload_bits: np.ndarray,
    qam_config: QAMConfig,
    ofdm_config: OFDMConfig,
) -> tuple[np.ndarray, np.ndarray, int]:
    _, data_indices = _subcarrier_layout(ofdm_config)
    bits_per_frame = (
        len(data_indices) * qam_config.bits_per_symbol
    )
    number_of_frames = math.ceil(len(payload_bits) / bits_per_frame)
    padded_length = number_of_frames * bits_per_frame
    padded_bits = np.pad(
        payload_bits,
        (0, padded_length - len(payload_bits)),
    )
    metadata: dict[str, object] = {
        "sample_rate": ofdm_config.sample_rate,
        "qam_order": qam_config.order,
    }

    qam_stream = ProcessorStage(
        BitsToQAMSymbols(qam_config.bits_per_symbol)
    ).process(
        _frame_bit_source(
            padded_bits,
            bits_per_frame,
            metadata,
        )
    )
    ofdm_stream = QAMSymbolsToOFDMFrame(ofdm_config).process(qam_stream)
    frames = [
        np.asarray(block.data, dtype=np.complex128)
        async for block in ofdm_stream
    ]
    if len(frames) != number_of_frames:
        raise RuntimeError("The OFDM modulator produced an unexpected frame count")

    iq_payload = np.concatenate(frames)
    audio_blocks = [
        block
        async for block in IQSymbolsToAudioSamples(
            _make_iq_config(ofdm_config)
        ).process(_single_iq_block(iq_payload, metadata))
    ]
    if len(audio_blocks) != 1:
        raise RuntimeError("The IQ modulator produced an unexpected block count")

    audio = np.asarray(audio_blocks[0].data, dtype=np.float64)
    peak = float(np.max(np.abs(audio)))
    if peak <= 0:
        raise RuntimeError("The OFDM modulator produced a silent signal")
    audio *= OFDM_PEAK_AMPLITUDE / peak
    return audio, padded_bits.astype(np.uint8), number_of_frames


async def _framed_audio(
    payload_audio: np.ndarray,
    preamble: np.ndarray,
    sample_rate: int,
) -> Stream[AudioSampleBlock]:
    leading_silence = np.zeros(
        round(LEADING_SILENCE_SECONDS * sample_rate),
        dtype=np.float64,
    )
    guard = np.zeros(
        round(GUARD_DURATION_SECONDS * sample_rate),
        dtype=np.float64,
    )
    trailing_silence = np.zeros(
        round(TRAILING_SILENCE_SECONDS * sample_rate),
        dtype=np.float64,
    )
    yield AudioSampleBlock(
        data=np.concatenate(
            (
                leading_silence,
                preamble,
                guard,
                payload_audio,
                trailing_silence,
            )
        ),
        is_final=True,
        metadata={"sample_rate": sample_rate},
    )


async def _first_packet(
    receiver: WebserverDeviceLayerReceiverSource,
    *,
    preamble: np.ndarray,
    packet_num_samples: int,
) -> PacketAudioSampleBlock:
    packets = FindPackets(
        preamble=preamble,
        packet_num_samples=packet_num_samples,
        threshold=PREAMBLE_DETECTION_THRESHOLD,
    ).process(receiver.stream())
    async for packet in packets:
        return packet
    raise RuntimeError("No complete OFDM packet was received")


def _ideal_constellation(qam_config: QAMConfig) -> np.ndarray:
    return np.asarray(
        list(
            qam_gray_lookup_table(
                qam_config.bits_per_symbol
            ).values()
        ),
        dtype=np.complex128,
    )


def _rms_evm(
    symbols: np.ndarray,
    qam_config: QAMConfig,
) -> float:
    constellation = _ideal_constellation(qam_config)
    distances = np.abs(
        symbols[:, np.newaxis]
        - constellation[np.newaxis, :]
    )
    nearest = constellation[np.argmin(distances, axis=1)]
    error_power = float(np.mean(np.abs(symbols - nearest) ** 2))
    reference_power = float(np.mean(np.abs(nearest) ** 2))
    return math.sqrt(error_power / max(reference_power, 1e-15))


def _demodulate_frames(
    baseband: np.ndarray,
    ofdm_config: OFDMConfig,
    *,
    maximum_frames: int | None = None,
) -> np.ndarray:
    frame_samples = ofdm_config.get_samples_per_ofdm_frame()
    number_of_frames = len(baseband) // frame_samples
    if maximum_frames is not None:
        number_of_frames = min(number_of_frames, maximum_frames)
    if number_of_frames <= 0:
        raise ValueError("The recording contains no complete OFDM frame")

    demodulator = IQSymbolsToQAMSymbols(ofdm_config)
    frames = [
        demodulator.ofdm_demodulate_frame(
            baseband[
                index * frame_samples:
                (index + 1) * frame_samples
            ]
        )
        for index in range(number_of_frames)
    ]
    symbols = np.concatenate(frames)
    if not np.all(np.isfinite(symbols)):
        raise FloatingPointError(
            "OFDM equalization produced non-finite symbols"
        )
    return symbols


def _find_timing_offset(
    baseband: np.ndarray,
    *,
    nominal_start: int,
    payload_samples: int,
    qam_config: QAMConfig,
    ofdm_config: OFDMConfig,
) -> int:
    radius = min(
        256,
        max(16, ofdm_config.cp_len // 2),
        nominal_start,
        len(baseband) - nominal_start - payload_samples,
    )
    if radius < 0:
        raise ValueError("The received packet is shorter than expected")

    def score(offset: int) -> float:
        start = nominal_start + offset
        stop = start + payload_samples
        try:
            symbols = _demodulate_frames(
                baseband[start:stop],
                ofdm_config,
                maximum_frames=2,
            )
            return _rms_evm(symbols, qam_config)
        except (ValueError, FloatingPointError):
            return math.inf

    coarse_offsets = range(-radius, radius + 1, 8)
    best_coarse = min(coarse_offsets, key=score)
    refinement = range(
        max(-radius, best_coarse - 7),
        min(radius, best_coarse + 7) + 1,
    )
    best_offset = min(refinement, key=score)
    if not math.isfinite(score(best_offset)):
        raise RuntimeError("Could not find a valid OFDM frame boundary")
    return best_offset


def _decode_packet(
    packet: PacketAudioSampleBlock,
    *,
    payload_samples: int,
    payload_bits: np.ndarray,
    qam_config: QAMConfig,
    ofdm_config: OFDMConfig,
) -> tuple[np.ndarray, np.ndarray, int, float]:
    baseband = iq_demodulate(
        np.asarray(packet.data, dtype=np.float64),
        carrier_frequency=ofdm_config.center_frequency,
        sample_rate=ofdm_config.sample_rate,
        lowpass_cutoff=_make_iq_config(ofdm_config).lowpass_cutoff,
        numtaps=IQ_FILTER_TAPS,
        carrier_phase_origin=0,
    )
    nominal_start = round(
        GUARD_DURATION_SECONDS * ofdm_config.sample_rate
    )
    timing_offset = _find_timing_offset(
        baseband,
        nominal_start=nominal_start,
        payload_samples=payload_samples,
        qam_config=qam_config,
        ofdm_config=ofdm_config,
    )
    start = nominal_start + timing_offset
    equalized_symbols = _demodulate_frames(
        baseband[start:start + payload_samples],
        ofdm_config,
    )
    received_bits = qam_to_bits(
        equalized_symbols,
        qam_config.bits_per_symbol,
    )[: len(payload_bits)]
    return (
        received_bits,
        equalized_symbols,
        timing_offset,
        _rms_evm(equalized_symbols, qam_config),
    )


def _count_bit_errors(
    expected: np.ndarray,
    actual: np.ndarray,
) -> int:
    comparable = min(len(expected), len(actual))
    return int(
        np.count_nonzero(
            expected[:comparable] != actual[:comparable]
        )
        + abs(len(expected) - len(actual))
    )


def _plot_constellation(result: OFDMQAMExerciseResult) -> None:
    symbols = result.equalized_symbols
    ideal = _ideal_constellation(result.qam_config)
    fig, axis = plt.subplots(
        figsize=(7, 7),
        constrained_layout=True,
    )
    axis.scatter(
        symbols.real,
        symbols.imag,
        s=14,
        alpha=0.55,
        label="Received and equalized",
    )
    axis.scatter(
        ideal.real,
        ideal.imag,
        marker="x",
        color="black",
        s=45,
        linewidths=1.2,
        label="Ideal constellation",
    )
    limit = max(
        1.25,
        float(np.quantile(np.abs(symbols), 0.995)) * 1.1,
    )
    axis.set_xlim(-limit, limit)
    axis.set_ylim(-limit, limit)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("Equalized in-phase component")
    axis.set_ylabel("Equalized quadrature component")
    axis.set_title(
        f"Received {result.qam_config.order}-QAM constellation"
    )
    axis.grid(True, alpha=0.25)
    axis.legend()
    plt.show()


def print_result(result: OFDMQAMExerciseResult) -> None:
    occupied_bandwidth = _occupied_baseband_bandwidth(
        result.ofdm_config
    )
    print("\nOFDM/QAM exercise result")
    print(f"  QAM order:                 {result.qam_config.order}-QAM")
    print(
        "  Bits per QAM symbol:       "
        f"{result.qam_config.bits_per_symbol}"
    )
    print(
        "  FFT / cyclic prefix:       "
        f"{result.ofdm_config.n_fft} / "
        f"{result.ofdm_config.cp_len} samples"
    )
    print(
        "  Positive subcarriers:      "
        f"{result.ofdm_config.num_positive_subcarriers}"
    )
    print(
        "  Pilot spacing:             "
        f"{result.ofdm_config.pilot_spacing}"
    )
    print(
        "  Data / pilot bins:         "
        f"{result.data_subcarriers_per_frame} / "
        f"{result.pilot_subcarriers_per_frame} per frame"
    )
    print(f"  Number of OFDM frames:     {result.number_of_frames}")
    print(f"  Occupied baseband width:   {occupied_bandwidth:.1f} Hz")
    print(
        "  Timing correction:         "
        f"{result.timing_offset_samples:+d} samples"
    )
    print(f"  Preamble score:            {result.preamble_score:.3f}")
    print(f"  RMS EVM:                   {result.rms_evm:.4f}")
    print(
        "  Bit errors:                "
        f"{result.bit_errors} / {len(result.transmitted_bits)}"
    )
    print(f"  Bit error rate:            {result.bit_error_rate:.6f}")


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
            name="ofdm-qam-exercise-webserver",
        )

    def _run(self) -> None:
        try:
            self.server.run()
        except BaseException as error:
            self.error = error

    async def start(self) -> None:
        _print_receiver_url(self.port)
        try:
            response = requests.get(
                f"https://127.0.0.1:{self.port}/api/status",
                timeout=1.0,
                verify=str(cert_path),
            )
            response.raise_for_status()
        except requests.RequestException:
            pass
        else:
            print(
                "Reusing the trAACer webserver that is already running "
                f"on port {self.port}."
            )
            return

        self.thread.start()
        deadline = time.monotonic() + SERVER_START_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            if self.server.started:
                return
            if not self.thread.is_alive():
                if self.error is not None:
                    raise RuntimeError(
                        "The receiver webserver failed"
                    ) from self.error
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
) -> None:
    deadline = time.monotonic() + SERVER_START_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if sender._try_start_waiting() is not None:
            return
        await asyncio.sleep(0.1)
    raise TimeoutError(
        "Timed out while preparing the browser microphone session"
    )


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


async def run_experiment(
    *,
    qam_config: QAMConfig = QAMConfig(),
    ofdm_config: OFDMConfig | None = None,
    payload_seed: int = PAYLOAD_SEED,
    show_plot: bool = True,
) -> OFDMQAMExerciseResult:
    """Transmit one configurable OFDM/QAM probe over the browser channel."""

    if ofdm_config is None:
        ofdm_config = OFDMConfig(
            center_frequency=12_000,
            sample_rate=48_000,
            cp_len=256,
            n_fft=1_024,
            pilot_spacing=8,
            pilot_value=1.0 + 0.0j,
            num_positive_subcarriers=32,
        )
    _validate_configuration(qam_config, ofdm_config)

    payload_bits = fixed_payload(payload_seed)
    payload_audio, padded_bits, number_of_frames = await _modulate_payload(
        payload_bits,
        qam_config,
        ofdm_config,
    )
    preamble = _create_preamble(ofdm_config.sample_rate)
    guard_samples = round(
        GUARD_DURATION_SECONDS * ofdm_config.sample_rate
    )
    trailing_samples = round(
        TRAILING_SILENCE_SECONDS * ofdm_config.sample_rate
    )
    packet_num_samples = (
        guard_samples + len(payload_audio) + trailing_samples
    )
    pilot_indices, data_indices = _subcarrier_layout(ofdm_config)

    print(
        f"Transmitting {len(payload_bits)} reproducible bits using "
        f"{qam_config.order}-QAM in {number_of_frames} OFDM frames."
    )

    webserver = _ManagedWebserver()
    sender: WebserverDeviceLayerSenderSink | None = None
    packet_task: asyncio.Task[PacketAudioSampleBlock] | None = None

    try:
        await webserver.start()
        server_url = f"https://127.0.0.1:{SERVER_PORT}"
        sender = WebserverDeviceLayerSenderSink(
            server_url=server_url,
            sample_rate=ofdm_config.sample_rate,
            cert_path=cert_path,
        )
        receiver = WebserverDeviceLayerReceiverSource(
            server_url=server_url,
            target_sample_rate=ofdm_config.sample_rate,
            cert_path=cert_path,
        )

        await _start_waiting_for_microphone(sender)
        print("Open the receiver URL shown above and start the microphone.")
        await _wait_for_status(
            sender,
            lambda status: bool(
                status.get("device_ready") and status.get("sample_rate")
            ),
            timeout_seconds=MICROPHONE_TIMEOUT_SECONDS,
            timeout_message="Timed out waiting for the browser microphone",
        )

        packet_task = asyncio.create_task(
            _first_packet(
                receiver,
                preamble=preamble,
                packet_num_samples=packet_num_samples,
            )
        )
        await _wait_for_status(
            sender,
            lambda status: status.get("download_active") is True,
            timeout_seconds=SERVER_START_TIMEOUT_SECONDS,
            timeout_message="Timed out while connecting the audio receiver",
        )
        await asyncio.wait_for(
            sender.play(
                _framed_audio(
                    payload_audio,
                    preamble,
                    ofdm_config.sample_rate,
                )
            ),
            timeout=RECEIVE_TIMEOUT_SECONDS,
        )
        packet = await asyncio.wait_for(
            packet_task,
            timeout=RECEIVE_TIMEOUT_SECONDS,
        )

        (
            received_bits,
            equalized_symbols,
            timing_offset,
            rms_evm,
        ) = _decode_packet(
            packet,
            payload_samples=len(payload_audio),
            payload_bits=padded_bits,
            qam_config=qam_config,
            ofdm_config=ofdm_config,
        )
        received_payload = received_bits[: len(payload_bits)]
        result = OFDMQAMExerciseResult(
            qam_config=qam_config,
            ofdm_config=ofdm_config,
            transmitted_bits=payload_bits,
            received_bits=received_payload,
            equalized_symbols=equalized_symbols,
            bit_errors=_count_bit_errors(
                payload_bits,
                received_payload,
            ),
            rms_evm=rms_evm,
            timing_offset_samples=timing_offset,
            preamble_score=packet.correlation_score,
            number_of_frames=number_of_frames,
            data_subcarriers_per_frame=len(data_indices),
            pilot_subcarriers_per_frame=len(pilot_indices),
        )
        print_result(result)
        if show_plot:
            _plot_constellation(result)
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
    asyncio.run(run_experiment())
