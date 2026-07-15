from __future__ import annotations

import argparse
import asyncio
import math
import threading
import time
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy.signal import correlate, find_peaks

from traacer.network_stack.layer.base import (
    AudioSampleBlock,
    BitBlock,
    DataBlock,
    ProcessorStage,
    Stage,
    Stream,
    run_webserver,
)
from traacer.network_stack.layer.device_layer.WebserverDeviceLayer import (
    WebserverDeviceLayerReceiverSource,
    WebserverDeviceLayerSenderSink,
)
from traacer.network_stack.layer.physical_layer.error_correction import (
    AddGuardSilence,
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
    QAMSymbolBlock,
    QAMSymbolsToOFDMFrame,
    get_used_indices,
    qam_gray_lookup_table,
    qam_to_bits,
)
from traacer.network_stack.layer.physical_layer.synchronization import (
    PrependPreamble,
    create_chirp_preamble,
)
from traacer.receiver.server import cert_path


SAMPLE_RATE = 48_000
CENTER_FREQUENCY_HZ = 12_000
N_FFT = 4_096
CYCLIC_PREFIX_SAMPLES = 1_024
PILOT_SPACING = 8
PILOT_VALUE = 1.0 + 0.0j
QAM_BITS_PER_SYMBOL = 4
POSITIVE_SUBCARRIER_COUNTS = (8, 16, 32, 64, 128, 256, 512)
LOWPASS_CUTOFF_HZ = 10_000.0
LOWPASS_NUMTAPS = 257
OFDM_PEAK_AMPLITUDE = 0.7

# Every selected carrier count provides a power-of-two multiple of 14 data bins.
# 896 QAM symbols therefore fill an integer number of complete frames for every
# run, so BER and signal-space plots never include padding symbols.
QAM_SYMBOLS_PER_RUN = 896
BIT_PATTERN_LENGTH = QAM_SYMBOLS_PER_RUN * QAM_BITS_PER_SYMBOL
BIT_PATTERN_SEED = 0x0FD5
BIT_PATTERN = np.random.default_rng(BIT_PATTERN_SEED).integers(
    0,
    2,
    size=BIT_PATTERN_LENGTH,
    dtype=np.uint8,
)

PREAMBLE_START_FREQUENCY_HZ = 500.0
PREAMBLE_END_FREQUENCY_HZ = 16_000.0
PREAMBLE_DURATION_SECONDS = 0.08
PREAMBLE_AMPLITUDE = 0.8
GUARD_DURATION_SECONDS = 0.08
LEADING_SILENCE_SECONDS = 0.5
TRAILING_SILENCE_SECONDS = 0.5
DETECTION_THRESHOLD = 0.20
MEASUREMENT_TIMEOUT_SECONDS = 120.0
CSV_OUTPUT_PATH = Path("measurements/ofdm_subcarrier_ber.csv")


@dataclass(frozen=True, slots=True)
class OFDMSubcarrierRunResult:
    run_index: int
    num_positive_subcarriers: int
    total_used_subcarriers: int
    pilot_subcarriers_per_frame: int
    data_subcarriers_per_frame: int
    num_ofdm_frames: int
    occupied_baseband_bandwidth_hz: float
    compared_bits: int
    bit_errors: int
    ber: float
    rms_evm: float
    timing_offset_samples: int
    preamble_score: float
    preamble_start_sample: int


@dataclass(frozen=True, slots=True)
class OFDMSubcarrierBERResult:
    sample_rate: int
    center_frequency_hz: int
    qam_bits_per_symbol: int
    bit_pattern: np.ndarray
    runs: tuple[OFDMSubcarrierRunResult, ...]


@dataclass(frozen=True, slots=True)
class OFDMSubcarrierBERBlock(DataBlock[OFDMSubcarrierBERResult]):
    pass


def create_experiment_preamble(sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    if PREAMBLE_END_FREQUENCY_HZ >= sample_rate / 2:
        raise ValueError("The preamble must remain below the Nyquist frequency")
    return PREAMBLE_AMPLITUDE * create_chirp_preamble(
        start_frequency=PREAMBLE_START_FREQUENCY_HZ,
        end_frequency=PREAMBLE_END_FREQUENCY_HZ,
        duration_in_sec=PREAMBLE_DURATION_SECONDS,
        sampling_frequency=sample_rate,
    )


def _normalized_correlation(
    samples: np.ndarray,
    reference: np.ndarray,
) -> np.ndarray:
    correlation = correlate(samples, reference, mode="valid", method="fft")
    reference_energy = float(np.linalg.norm(reference))
    cumulative_energy = np.concatenate(
        ([0.0], np.cumsum(np.square(samples, dtype=np.float64)))
    )
    window_energy = np.sqrt(
        np.maximum(
            0.0,
            cumulative_energy[len(reference):]
            - cumulative_energy[:-len(reference)],
        )
    )
    normalized = np.zeros_like(correlation, dtype=np.float64)
    non_silent = window_energy > reference_energy * 1e-12
    normalized[non_silent] = correlation[non_silent] / (
        window_energy[non_silent] * reference_energy
    )
    return normalized


def _subcarrier_layout(config: OFDMConfig) -> tuple[np.ndarray, np.ndarray]:
    used_indices = get_used_indices(
        config.n_fft,
        config.num_positive_subcarriers,
    )
    pilot_indices = used_indices[::config.pilot_spacing]
    data_indices = np.setdiff1d(used_indices, pilot_indices)
    return pilot_indices, data_indices


def _make_ofdm_config(
    num_positive_subcarriers: int,
    *,
    center_frequency_hz: int,
    sample_rate: int,
) -> OFDMConfig:
    return OFDMConfig(
        center_frequency=center_frequency_hz,
        sample_rate=sample_rate,
        cp_len=CYCLIC_PREFIX_SAMPLES,
        n_fft=N_FFT,
        pilot_spacing=PILOT_SPACING,
        pilot_value=PILOT_VALUE,
        num_positive_subcarriers=num_positive_subcarriers,
    )


def _make_iq_config(
    *,
    center_frequency_hz: int,
    sample_rate: int,
) -> IQConfig:
    return IQConfig(
        carrier_frequency=center_frequency_hz,
        sample_rate=sample_rate,
        lowpass_cutoff=LOWPASS_CUTOFF_HZ,
        numtaps=LOWPASS_NUMTAPS,
        carrier_phase_origin=0,
    )


def _nearest_constellation_metrics(
    symbols: np.ndarray,
) -> tuple[np.ndarray, float]:
    lookup = qam_gray_lookup_table(QAM_BITS_PER_SYMBOL)
    constellation = np.asarray(list(lookup.values()), dtype=np.complex128)
    distances = np.abs(symbols[:, None] - constellation[None, :])
    nearest = constellation[np.argmin(distances, axis=1)]
    error_power = float(np.mean(np.abs(symbols - nearest) ** 2))
    reference_power = float(np.mean(np.abs(nearest) ** 2))
    rms_evm = math.sqrt(error_power / max(reference_power, 1e-15))
    return nearest, rms_evm


class OFDMSignalSpacePlot(Stage[QAMSymbolBlock, QAMSymbolBlock]):
    """Plot equalized QAM clusters for each positive-subcarrier count."""

    def __init__(
        self,
        positive_subcarrier_counts: tuple[int, ...] = POSITIVE_SUBCARRIER_COUNTS,
        *,
        max_points_per_run: int = 2_000,
    ) -> None:
        if not positive_subcarrier_counts:
            raise ValueError("At least one subcarrier count is required")
        if max_points_per_run < 2:
            raise ValueError("max_points_per_run must be at least 2")
        self.positive_subcarrier_counts = tuple(positive_subcarrier_counts)
        self.max_points_per_run = max_points_per_run
        self._figure: Figure | None = None
        self._axes_by_count: dict[int, Axes] = {}
        self._plotted_counts: set[int] = set()
        self._colorbar_created = False

    async def process(
        self,
        stream: Stream[QAMSymbolBlock],
    ) -> Stream[QAMSymbolBlock]:
        async for block in stream:
            count = block.metadata.get("num_positive_subcarriers")
            if not isinstance(count, int):
                raise ValueError(
                    "OFDMSignalSpacePlot requires integer metadata "
                    "'num_positive_subcarriers'"
                )
            self._update_plot(block, count)
            yield block

    def _initialize_plot(self) -> None:
        columns = min(3, len(self.positive_subcarrier_counts))
        rows = math.ceil(len(self.positive_subcarrier_counts) / columns)
        self._figure, axes = plt.subplots(
            rows,
            columns,
            figsize=(4.5 * columns, 4.0 * rows),
            constrained_layout=True,
            squeeze=False,
        )
        flat_axes = axes.ravel()
        for count, axis in zip(self.positive_subcarrier_counts, flat_axes):
            self._axes_by_count[count] = axis
            axis.set_title(f"{count} positive subcarriers (waiting)")
            axis.set_xlabel("Equalized in-phase")
            axis.set_ylabel("Equalized quadrature")
            axis.grid(True, alpha=0.25)
        for axis in flat_axes[len(self.positive_subcarrier_counts):]:
            axis.set_visible(False)
        self._figure.suptitle(
            "OFDM 16-QAM signal space — color progresses through transmission"
        )

    def _update_plot(self, block: QAMSymbolBlock, count: int) -> None:
        if self._figure is None:
            self._initialize_plot()
        assert self._figure is not None
        if count not in self._axes_by_count:
            raise ValueError(f"Unexpected positive-subcarrier count: {count}")

        symbols = np.asarray(block.data, dtype=np.complex128).reshape(-1)
        time_position = np.linspace(0.0, 1.0, len(symbols))
        if len(symbols) > self.max_points_per_run:
            indices = np.linspace(
                0,
                len(symbols) - 1,
                self.max_points_per_run,
                dtype=int,
            )
            symbols = symbols[indices]
            time_position = time_position[indices]

        axis = self._axes_by_count[count]
        axis.clear()
        scatter = axis.scatter(
            symbols.real,
            symbols.imag,
            c=time_position,
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
            s=15,
            alpha=0.7,
            linewidths=0.0,
            zorder=3,
        )
        ideal = np.asarray(
            list(qam_gray_lookup_table(QAM_BITS_PER_SYMBOL).values()),
            dtype=np.complex128,
        )
        axis.scatter(
            ideal.real,
            ideal.imag,
            marker="x",
            color="black",
            s=30,
            linewidths=0.9,
            label="Ideal 16-QAM",
            zorder=1,
        )
        limit = max(1.25, float(np.quantile(np.abs(symbols), 0.995)) * 1.1)
        axis.set_xlim(-limit, limit)
        axis.set_ylim(-limit, limit)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel("Equalized in-phase")
        axis.set_ylabel("Equalized quadrature")
        axis.set_title(
            f"{count} positive subcarriers — {len(block.data)} QAM symbols"
        )
        axis.grid(True, alpha=0.25)
        axis.legend(loc="upper right", fontsize=7)

        if not self._colorbar_created:
            self._figure.colorbar(
                scatter,
                ax=list(self._axes_by_count.values()),
                label="Position within run (early → late)",
                shrink=0.8,
            )
            self._colorbar_created = True

        self._plotted_counts.add(count)
        self._figure.canvas.draw()
        if self._plotted_counts == set(self.positive_subcarrier_counts):
            plt.show(block=False)


@dataclass(frozen=True, slots=True)
class _LocatedPacket:
    run_index: int
    num_positive_subcarriers: int
    preamble_start: int
    nominal_packet_start: int
    packet_samples: int
    preamble_score: float


class MeasureOFDMSubcarrierBER(
    Stage[AudioSampleBlock, OFDMSubcarrierBERBlock]
):
    def __init__(
        self,
        *,
        center_frequency_hz: int = CENTER_FREQUENCY_HZ,
        positive_subcarrier_counts: tuple[int, ...] = POSITIVE_SUBCARRIER_COUNTS,
        bit_pattern: np.ndarray = BIT_PATTERN,
        preamble: np.ndarray | None = None,
        guard_duration_seconds: float = GUARD_DURATION_SECONDS,
        detection_threshold: float = DETECTION_THRESHOLD,
        sample_rate: int = SAMPLE_RATE,
        max_capture_seconds: float = MEASUREMENT_TIMEOUT_SECONDS,
        signal_space_plot: OFDMSignalSpacePlot | None = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.center_frequency_hz = center_frequency_hz
        self.positive_subcarrier_counts = tuple(positive_subcarrier_counts)
        self.bit_pattern = np.asarray(bit_pattern, dtype=np.uint8).reshape(-1)
        self.preamble = np.asarray(
            create_experiment_preamble(sample_rate)
            if preamble is None
            else preamble,
            dtype=np.float64,
        ).reshape(-1)
        self.guard_samples = round(guard_duration_seconds * sample_rate)
        self.detection_threshold = detection_threshold
        self.max_capture_samples = round(max_capture_seconds * sample_rate)
        self.signal_space_plot = signal_space_plot

        self._buffer = np.empty(0, dtype=np.float64)
        self._absolute_offset = 0
        self._samples_at_last_attempt = 0
        self._analysis_interval_samples = max(1, round(0.25 * sample_rate))
        self._validate_configuration()

    def _validate_configuration(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if not 0 < self.center_frequency_hz < self.sample_rate / 2:
            raise ValueError("center_frequency_hz must be below Nyquist")
        if not self.positive_subcarrier_counts:
            raise ValueError("At least one positive-subcarrier count is required")
        if any(
            later <= earlier
            for earlier, later in zip(
                self.positive_subcarrier_counts,
                self.positive_subcarrier_counts[1:],
            )
        ):
            raise ValueError("Positive-subcarrier counts must strictly increase")
        if len(self.preamble) == 0:
            raise ValueError("preamble cannot be empty")
        if self.guard_samples < 0:
            raise ValueError("guard_duration_seconds cannot be negative")
        if not 0.0 < self.detection_threshold <= 1.0:
            raise ValueError("detection_threshold must be in (0, 1]")
        if len(self.bit_pattern) == 0 or np.any(
            (self.bit_pattern != 0) & (self.bit_pattern != 1)
        ):
            raise ValueError("bit_pattern must contain only 0 and 1")
        if len(self.bit_pattern) % QAM_BITS_PER_SYMBOL:
            raise ValueError("bit_pattern must contain complete QAM symbols")

        for count in self.positive_subcarrier_counts:
            self._validate_subcarrier_count(count)

    def _validate_subcarrier_count(self, count: int) -> None:
        if count < PILOT_SPACING or count >= N_FFT // 2:
            raise ValueError(
                f"Positive-subcarrier count must be in [{PILOT_SPACING}, "
                f"{N_FFT // 2})"
            )
        occupied_bandwidth = count * self.sample_rate / N_FFT
        if occupied_bandwidth >= LOWPASS_CUTOFF_HZ:
            raise ValueError(
                f"{count} subcarriers exceed the IQ low-pass bandwidth"
            )
        if (
            self.center_frequency_hz - occupied_bandwidth <= 0
            or self.center_frequency_hz + occupied_bandwidth
            >= self.sample_rate / 2
        ):
            raise ValueError(
                f"{count} subcarriers do not fit around the center frequency"
            )

        config = _make_ofdm_config(
            count,
            center_frequency_hz=self.center_frequency_hz,
            sample_rate=self.sample_rate,
        )
        _, data_indices = _subcarrier_layout(config)
        qam_symbols = len(self.bit_pattern) // QAM_BITS_PER_SYMBOL
        if qam_symbols % len(data_indices):
            raise ValueError(
                f"The bit pattern does not fill complete frames for count {count}"
            )

    async def process(
        self,
        stream: Stream[AudioSampleBlock],
    ) -> Stream[OFDMSubcarrierBERBlock]:
        async for block in stream:
            received_sample_rate = int(
                block.metadata.get("sample_rate") or self.sample_rate
            )
            if received_sample_rate != self.sample_rate:
                raise ValueError(
                    "Received sample rate does not match the experiment rate: "
                    f"{received_sample_rate} != {self.sample_rate}"
                )
            samples = np.asarray(block.data, dtype=np.float64).reshape(-1)
            if len(samples):
                self._buffer = np.concatenate((self._buffer, samples))
                self._trim_buffer()

            should_analyze = (
                len(self._buffer) - self._samples_at_last_attempt
                >= self._analysis_interval_samples
                or block.is_final
            )
            if not should_analyze:
                continue
            self._samples_at_last_attempt = len(self._buffer)

            packets = self._try_locate_packets()
            if packets is None:
                continue
            result = await self._measure_packets(packets)
            yield OFDMSubcarrierBERBlock(
                data=result,
                is_final=True,
                metadata={"sample_rate": self.sample_rate},
            )
            return

        raise RuntimeError("The audio stream ended before all OFDM runs arrived")

    def _trim_buffer(self) -> None:
        if len(self._buffer) <= self.max_capture_samples:
            return
        dropped = len(self._buffer) - self.max_capture_samples
        self._buffer = self._buffer[dropped:]
        self._absolute_offset += dropped
        self._samples_at_last_attempt = max(
            0,
            self._samples_at_last_attempt - dropped,
        )

    def _packet_samples(self, count: int) -> int:
        config = _make_ofdm_config(
            count,
            center_frequency_hz=self.center_frequency_hz,
            sample_rate=self.sample_rate,
        )
        _, data_indices = _subcarrier_layout(config)
        qam_symbols = len(self.bit_pattern) // QAM_BITS_PER_SYMBOL
        num_frames = qam_symbols // len(data_indices)
        return num_frames * config.get_samples_per_ofdm_frame()

    def _try_locate_packets(self) -> tuple[_LocatedPacket, ...] | None:
        if len(self._buffer) < len(self.preamble):
            return None
        scores = np.abs(_normalized_correlation(self._buffer, self.preamble))
        shortest_packet = self._packet_samples(
            self.positive_subcarrier_counts[-1]
        )
        shortest_frame = (
            len(self.preamble) + 2 * self.guard_samples + shortest_packet
        )
        peaks, _ = find_peaks(
            scores,
            height=self.detection_threshold,
            distance=max(len(self.preamble), shortest_frame // 2),
        )
        if len(peaks) < len(self.positive_subcarrier_counts):
            return None

        packets = []
        for run_index, (preamble_start, count) in enumerate(
            zip(
                peaks[:len(self.positive_subcarrier_counts)],
                self.positive_subcarrier_counts,
            ),
            start=1,
        ):
            nominal_start = (
                int(preamble_start) + len(self.preamble) + self.guard_samples
            )
            packet_samples = self._packet_samples(count)
            timing_radius = CYCLIC_PREFIX_SAMPLES // 2
            if nominal_start + packet_samples + timing_radius > len(self._buffer):
                return None
            packets.append(
                _LocatedPacket(
                    run_index=run_index,
                    num_positive_subcarriers=count,
                    preamble_start=int(preamble_start),
                    nominal_packet_start=nominal_start,
                    packet_samples=packet_samples,
                    preamble_score=float(scores[preamble_start]),
                )
            )
        return tuple(packets)

    async def _measure_packets(
        self,
        packets: tuple[_LocatedPacket, ...],
    ) -> OFDMSubcarrierBERResult:
        runs = []
        for packet in packets:
            runs.append(await self._measure_packet(packet))
        return OFDMSubcarrierBERResult(
            sample_rate=self.sample_rate,
            center_frequency_hz=self.center_frequency_hz,
            qam_bits_per_symbol=QAM_BITS_PER_SYMBOL,
            bit_pattern=self.bit_pattern.copy(),
            runs=tuple(runs),
        )

    async def _measure_packet(
        self,
        packet: _LocatedPacket,
    ) -> OFDMSubcarrierRunResult:
        config = _make_ofdm_config(
            packet.num_positive_subcarriers,
            center_frequency_hz=self.center_frequency_hz,
            sample_rate=self.sample_rate,
        )
        timing_offset = self._find_timing_offset(packet, config)
        packet_start = packet.nominal_packet_start + timing_offset
        packet_audio = self._buffer[
            packet_start:packet_start + packet.packet_samples
        ]
        equalized_symbols = self._demodulate_packet(packet_audio, config)

        qam_block = QAMSymbolBlock(
            data=equalized_symbols,
            is_final=True,
            metadata={
                "num_positive_subcarriers": packet.num_positive_subcarriers,
                "sample_rate": self.sample_rate,
            },
        )
        symbol_stream = self._single_qam_block(qam_block)
        if self.signal_space_plot is not None:
            symbol_stream = self.signal_space_plot.process(symbol_stream)
        plotted_block = await anext(symbol_stream)
        await symbol_stream.aclose()

        received_bits = qam_to_bits(
            np.asarray(plotted_block.data, dtype=np.complex128),
            QAM_BITS_PER_SYMBOL,
        )
        received_bits = received_bits[:len(self.bit_pattern)]
        bit_errors = int(np.count_nonzero(received_bits != self.bit_pattern))
        _, rms_evm = _nearest_constellation_metrics(equalized_symbols)
        pilot_indices, data_indices = _subcarrier_layout(config)
        num_frames = len(equalized_symbols) // len(data_indices)

        return OFDMSubcarrierRunResult(
            run_index=packet.run_index,
            num_positive_subcarriers=packet.num_positive_subcarriers,
            total_used_subcarriers=(
                2 * packet.num_positive_subcarriers
            ),
            pilot_subcarriers_per_frame=len(pilot_indices),
            data_subcarriers_per_frame=len(data_indices),
            num_ofdm_frames=num_frames,
            occupied_baseband_bandwidth_hz=(
                packet.num_positive_subcarriers
                * self.sample_rate
                / config.n_fft
            ),
            compared_bits=len(self.bit_pattern),
            bit_errors=bit_errors,
            ber=bit_errors / len(self.bit_pattern),
            rms_evm=rms_evm,
            timing_offset_samples=timing_offset,
            preamble_score=packet.preamble_score,
            preamble_start_sample=(
                self._absolute_offset + packet.preamble_start
            ),
        )

    def _find_timing_offset(
        self,
        packet: _LocatedPacket,
        config: OFDMConfig,
    ) -> int:
        frame_samples = config.get_samples_per_ofdm_frame()
        radius = CYCLIC_PREFIX_SAMPLES // 2
        coarse_offsets = range(-radius, radius + 1, 8)
        best_offset = min(
            coarse_offsets,
            key=lambda offset: self._timing_score(
                packet.nominal_packet_start + offset,
                frame_samples,
                config,
            ),
        )
        refinement = range(
            max(-radius, best_offset - 7),
            min(radius, best_offset + 7) + 1,
        )
        return min(
            refinement,
            key=lambda offset: self._timing_score(
                packet.nominal_packet_start + offset,
                frame_samples,
                config,
            ),
        )

    def _timing_score(
        self,
        start: int,
        frame_samples: int,
        config: OFDMConfig,
    ) -> float:
        if start < 0 or start + frame_samples > len(self._buffer):
            return math.inf
        first_frame_audio = self._buffer[start:start + frame_samples]
        try:
            symbols = self._demodulate_packet(first_frame_audio, config)
            _, rms_evm = _nearest_constellation_metrics(symbols)
            return rms_evm
        except (ValueError, FloatingPointError):
            return math.inf

    def _demodulate_packet(
        self,
        packet_audio: np.ndarray,
        config: OFDMConfig,
    ) -> np.ndarray:
        iq_config = _make_iq_config(
            center_frequency_hz=self.center_frequency_hz,
            sample_rate=self.sample_rate,
        )
        baseband = iq_demodulate(
            packet_audio,
            carrier_frequency=iq_config.carrier_frequency,
            sample_rate=iq_config.sample_rate,
            lowpass_cutoff=iq_config.lowpass_cutoff,
            numtaps=iq_config.numtaps,
            carrier_phase_origin=iq_config.carrier_phase_origin,
        )
        frame_samples = config.get_samples_per_ofdm_frame()
        if len(baseband) % frame_samples:
            raise ValueError("OFDM packet does not contain complete frames")

        demodulator = IQSymbolsToQAMSymbols(config)
        frames = []
        for start in range(0, len(baseband), frame_samples):
            frames.append(
                demodulator.ofdm_demodulate_frame(
                    baseband[start:start + frame_samples]
                )
            )
        return np.concatenate(frames)

    async def _single_qam_block(
        self,
        block: QAMSymbolBlock,
    ) -> Stream[QAMSymbolBlock]:
        yield block


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
    yield IQSymbolBlock(data=samples, is_final=True, metadata=metadata)


async def _single_audio_block(
    samples: np.ndarray,
    metadata: dict[str, object],
) -> Stream[AudioSampleBlock]:
    yield AudioSampleBlock(data=samples, is_final=True, metadata=metadata)


async def ofdm_subcarrier_probe_source(
    *,
    center_frequency_hz: int = CENTER_FREQUENCY_HZ,
    positive_subcarrier_counts: tuple[int, ...] = POSITIVE_SUBCARRIER_COUNTS,
    sample_rate: int = SAMPLE_RATE,
) -> Stream[AudioSampleBlock]:
    preamble = create_experiment_preamble(sample_rate)
    guard_samples = round(GUARD_DURATION_SECONDS * sample_rate)
    parts = [
        np.zeros(round(LEADING_SILENCE_SECONDS * sample_rate), dtype=np.float64)
    ]

    for count in positive_subcarrier_counts:
        config = _make_ofdm_config(
            count,
            center_frequency_hz=center_frequency_hz,
            sample_rate=sample_rate,
        )
        _, data_indices = _subcarrier_layout(config)
        metadata: dict[str, object] = {
            "sample_rate": sample_rate,
            "num_positive_subcarriers": count,
        }
        bit_stream = _frame_bit_source(
            BIT_PATTERN,
            len(data_indices) * QAM_BITS_PER_SYMBOL,
            metadata,
        )
        qam_stream = ProcessorStage(
            BitsToQAMSymbols(QAM_BITS_PER_SYMBOL)
        ).process(bit_stream)
        iq_frame_stream = QAMSymbolsToOFDMFrame(config).process(qam_stream)
        iq_frames = [
            np.asarray(block.data, dtype=np.complex128)
            async for block in iq_frame_stream
        ]
        iq_payload = np.concatenate(iq_frames)

        iq_config = _make_iq_config(
            center_frequency_hz=center_frequency_hz,
            sample_rate=sample_rate,
        )
        audio_blocks = [
            block
            async for block in IQSymbolsToAudioSamples(iq_config).process(
                _single_iq_block(iq_payload, metadata)
            )
        ]
        payload_audio = np.asarray(audio_blocks[0].data, dtype=np.float64)
        peak = float(np.max(np.abs(payload_audio)))
        if peak <= 0.0:
            raise ValueError("OFDM modulator produced a silent packet")
        payload_audio = payload_audio * (OFDM_PEAK_AMPLITUDE / peak)

        guarded = AddGuardSilence(guard_samples).process(
            _single_audio_block(payload_audio, metadata)
        )
        framed = PrependPreamble(preamble).process(guarded)
        async for block in framed:
            parts.append(np.asarray(block.data, dtype=np.float64))

    parts.append(
        np.zeros(round(TRAILING_SILENCE_SECONDS * sample_rate), dtype=np.float64)
    )
    yield AudioSampleBlock(
        data=np.concatenate(parts),
        is_final=True,
        metadata={
            "sample_rate": sample_rate,
            "center_frequency_hz": center_frequency_hz,
            "purpose": "ofdm_ber_positive_subcarrier_sweep",
        },
    )


def write_ofdm_subcarrier_csv(
    result: OFDMSubcarrierBERResult,
    path: str | Path = CSV_OUTPUT_PATH,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = np.asarray(
        [
            (
                run.run_index,
                result.center_frequency_hz,
                run.num_positive_subcarriers,
                run.total_used_subcarriers,
                run.pilot_subcarriers_per_frame,
                run.data_subcarriers_per_frame,
                run.num_ofdm_frames,
                run.occupied_baseband_bandwidth_hz,
                run.compared_bits,
                run.bit_errors,
                run.ber,
                run.rms_evm,
                run.timing_offset_samples,
                run.preamble_score,
                run.preamble_start_sample,
            )
            for run in result.runs
        ],
        dtype=np.float64,
    )
    np.savetxt(
        output_path,
        rows,
        delimiter=",",
        header=(
            "run_index,center_frequency_hz,num_positive_subcarriers,"
            "total_used_subcarriers,pilot_subcarriers_per_frame,"
            "data_subcarriers_per_frame,num_ofdm_frames,"
            "occupied_baseband_bandwidth_hz,compared_bits,bit_errors,ber,"
            "rms_evm,timing_offset_samples,preamble_score,"
            "preamble_start_sample"
        ),
        comments="",
        fmt=(
            "%d", "%d", "%d", "%d", "%d", "%d", "%d", "%.8f", "%d",
            "%d", "%.9f", "%.9f", "%d", "%.9f", "%d",
        ),
    )
    print(f"CSV written to {output_path}")


def plot_ofdm_subcarrier_ber(result: OFDMSubcarrierBERResult) -> None:
    counts = np.asarray(
        [run.num_positive_subcarriers for run in result.runs]
    )
    ber = np.asarray([run.ber for run in result.runs])
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    ax.plot(counts, ber, marker="o")
    ax.set_xscale("log", base=2)
    ax.set_xticks(counts, labels=[str(count) for count in counts])
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Number of positive OFDM subcarriers")
    ax.set_ylabel("Bit error rate")
    ax.set_title(
        f"OFDM BER versus positive subcarriers at "
        f"{result.center_frequency_hz:g} Hz"
    )
    ax.grid(True, which="both", alpha=0.3)
    plt.show()


def print_result(result: OFDMSubcarrierBERResult) -> None:
    print("\nOFDM positive-subcarrier sweep")
    print(f"  Center frequency: {result.center_frequency_hz:g} Hz")
    print(f"  FFT / CP:         {N_FFT} / {CYCLIC_PREFIX_SAMPLES} samples")
    print(f"  QAM:              {2 ** result.qam_bits_per_symbol}-QAM")
    print(f"  Pattern bits:     {len(result.bit_pattern)} per run")
    print("\n  run   +carriers   frames   errors        BER       RMS EVM")
    for run in result.runs:
        print(
            f"  {run.run_index:>3}   {run.num_positive_subcarriers:>9}"
            f"   {run.num_ofdm_frames:>6}"
            f"   {run.bit_errors:>4}/{run.compared_bits:<4}"
            f"   {run.ber:>9.6f}   {run.rms_evm:>8.4f}"
        )


async def _wait_for_microphone(
    sender: WebserverDeviceLayerSenderSink,
    timeout_seconds: float = 310.0,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            status = await asyncio.to_thread(sender._get_status)
        except Exception:
            await asyncio.sleep(0.1)
            continue
        if status.get("device_ready") and status.get("sample_rate"):
            return
        await asyncio.sleep(0.1)
    raise TimeoutError("Timed out waiting for the browser microphone")


async def main(
    *,
    center_frequency_hz: int = CENTER_FREQUENCY_HZ,
    positive_subcarrier_counts: tuple[int, ...] = POSITIVE_SUBCARRIER_COUNTS,
    csv_output_path: str | Path = CSV_OUTPUT_PATH,
    show_plot: bool = True,
) -> None:
    webserver_thread = threading.Thread(target=run_webserver, daemon=True)
    webserver_thread.start()

    sender = WebserverDeviceLayerSenderSink(
        server_url="https://127.0.0.1:8000",
        sample_rate=SAMPLE_RATE,
        cert_path=cert_path,
    )
    receiver = WebserverDeviceLayerReceiverSource(
        target_sample_rate=SAMPLE_RATE,
        cert_path=cert_path,
    )
    signal_space_plot = (
        OFDMSignalSpacePlot(positive_subcarrier_counts) if show_plot else None
    )
    measurement = MeasureOFDMSubcarrierBER(
        center_frequency_hz=center_frequency_hz,
        positive_subcarrier_counts=positive_subcarrier_counts,
        signal_space_plot=signal_space_plot,
    )
    transmit_task = asyncio.create_task(
        sender.consume(
            ofdm_subcarrier_probe_source(
                center_frequency_hz=center_frequency_hz,
                positive_subcarrier_counts=positive_subcarrier_counts,
            )
        )
    )

    try:
        await _wait_for_microphone(sender)

        async def receive_measurement() -> OFDMSubcarrierBERResult:
            async for block in measurement.process(receiver.stream()):
                return block.data
            raise RuntimeError("No OFDM subcarrier-sweep result was produced")

        result, _ = await asyncio.wait_for(
            asyncio.gather(receive_measurement(), transmit_task),
            timeout=MEASUREMENT_TIMEOUT_SECONDS,
        )
        print_result(result)
        write_ofdm_subcarrier_csv(result, csv_output_path)
        if show_plot:
            plot_ofdm_subcarrier_ber(result)
    finally:
        if not transmit_task.done():
            transmit_task.cancel()
        with suppress(asyncio.CancelledError, Exception):
            await transmit_task
        try:
            await asyncio.to_thread(sender._stop_blocking)
        except Exception:
            pass


def _parse_counts(value: str) -> tuple[int, ...]:
    try:
        counts = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Subcarrier counts must be comma-separated integers"
        ) from exc
    if not counts or any(
        later <= earlier for earlier, later in zip(counts, counts[1:])
    ):
        raise argparse.ArgumentTypeError(
            "Positive-subcarrier counts must strictly increase"
        )
    return counts


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure OFDM BER while increasing positive subcarriers."
    )
    parser.add_argument(
        "--center-frequency-hz",
        type=int,
        default=CENTER_FREQUENCY_HZ,
        help=f"OFDM center frequency (default: {CENTER_FREQUENCY_HZ} Hz)",
    )
    parser.add_argument(
        "--subcarriers",
        type=_parse_counts,
        default=POSITIVE_SUBCARRIER_COUNTS,
        help="Positive counts (default: 8,16,32,64,128,256,512)",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=CSV_OUTPUT_PATH,
        help=f"CSV output path (default: {CSV_OUTPUT_PATH})",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Disable both the OFDM signal-space and BER plots.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = _parse_args()
    asyncio.run(
        main(
            center_frequency_hz=arguments.center_frequency_hz,
            positive_subcarrier_counts=arguments.subcarriers,
            csv_output_path=arguments.csv,
            show_plot=not arguments.no_plot,
        )
    )
