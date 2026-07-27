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
from traacer.network_stack.layer.physical_layer.modulation.ask import (
    ASKAudioSamplesToBits,
    ASKConfig,
    BitsToASKAudioSamples,
)
from traacer.network_stack.layer.physical_layer.synchronization import (
    PacketAudioSampleBlock,
    PrependPreamble,
    create_chirp_preamble,
)
from traacer.receiver.server import cert_path


SAMPLE_RATE = 48_000
CARRIER_FREQUENCY_HZ = 8_000.0
SYMBOL_TIME_SECONDS = 5e-3
MODULATION_ORDERS = (2,)

# 840 is divisible by every bits-per-symbol value from 1 through 8. The seeded
# generator creates one fixed, repeatable pattern with broad symbol coverage at
# every tested ASK order.
BIT_PATTERN_LENGTH = 840
BIT_PATTERN_SEED = 0xA5C
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
ASK_AMPLITUDE = 0.8
DETECTION_THRESHOLD = 0.20
MEASUREMENT_TIMEOUT_SECONDS = 120.0
CSV_OUTPUT_PATH = Path("measurements/ask_order_ber-2.csv")
PLOT_OUTPUT_PATH = Path("measurements/ask-order-ber-2.svg")
SIGNAL_SPACE_PLOT_OUTPUT_PATH = Path(
    "measurements/ask-phase-drift.svg"
)


@dataclass(frozen=True, slots=True)
class ASKOrderRunResult:
    run_index: int
    modulation_order: int
    bits_per_symbol: int
    symbol_time_seconds: float
    samples_per_symbol: int
    compared_bits: int
    bit_errors: int
    ber: float
    estimated_channel_gain: float
    estimated_level_spacing: float
    timing_offset_samples: int
    preamble_score: float
    preamble_start_sample: int

    @property
    def symbol_rate_baud(self) -> float:
        return 1.0 / self.symbol_time_seconds

    @property
    def raw_bit_rate_bps(self) -> float:
        return self.bits_per_symbol * self.symbol_rate_baud


@dataclass(frozen=True, slots=True)
class ASKOrderBERResult:
    sample_rate: int
    carrier_frequency_hz: float
    bit_pattern: np.ndarray
    runs: tuple[ASKOrderRunResult, ...]


@dataclass(frozen=True, slots=True)
class ASKOrderBERBlock(DataBlock[ASKOrderBERResult]):
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


def _is_power_of_two(value: int) -> bool:
    return value >= 2 and value & (value - 1) == 0


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


def _ask_symbol_coordinates(
    samples: np.ndarray,
    config: ASKConfig,
) -> np.ndarray:
    """Return complex matched-filter coordinates for complete ASK symbols."""

    samples = np.asarray(samples, dtype=np.float64).reshape(-1)
    if len(samples) % config.samples_per_symbol:
        raise ValueError(
            "ASK audio does not contain a whole number of symbols"
        )

    symbols = samples.reshape(-1, config.samples_per_symbol)
    t = np.arange(config.samples_per_symbol, dtype=np.float64) / config.sample_rate
    angular_phase = 2.0 * np.pi * config.carrier_frequency * t
    window = np.ones(config.samples_per_symbol, dtype=np.float64)
    if config.ramp_samples:
        ramp = config.ramp_samples
        window[:ramp] *= np.linspace(0.0, 1.0, ramp, endpoint=False)
        window[-ramp:] *= np.linspace(1.0, 0.0, ramp, endpoint=False)

    sin_template = np.sin(angular_phase) * window
    cos_template = np.cos(angular_phase) * window
    sin_energy = float(np.dot(sin_template, sin_template))
    cos_energy = float(np.dot(cos_template, cos_template))
    if sin_energy <= 0.0 or cos_energy <= 0.0:
        raise ValueError("ASK correlation template has zero energy")

    in_phase = symbols @ sin_template / sin_energy
    quadrature = symbols @ cos_template / cos_energy
    return in_phase + 1j * quadrature


def _estimate_carrier_level(
    samples: np.ndarray,
    config: ASKConfig,
) -> float:
    """Estimate one ASK symbol amplitude without assuming its carrier phase."""

    if len(samples) != config.samples_per_symbol:
        raise ValueError(
            f"Expected {config.samples_per_symbol} samples, got {len(samples)}"
        )
    return float(np.abs(_ask_symbol_coordinates(samples, config)[0]))


def _make_ask_config(
    *,
    order: int,
    samples_per_symbol: int,
    carrier_frequency_hz: float,
    sample_rate: int,
) -> ASKConfig:
    return ASKConfig(
        sample_rate=sample_rate,
        symbol_rate=sample_rate / samples_per_symbol,
        carrier_frequency=carrier_frequency_hz,
        order=order,
        amplitude=ASK_AMPLITUDE,
        ramp_duration=min(2e-3, 0.1 * samples_per_symbol / sample_rate),
        continuous_phase=True,
    )


class ASKSignalSpacePlot(
    Stage[PacketAudioSampleBlock, PacketAudioSampleBlock]
):
    """Plot pilot-normalized ASK clusters, with color showing packet time."""

    def __init__(
        self,
        modulation_orders: tuple[int, ...] = MODULATION_ORDERS,
        *,
        max_points_per_order: int = 2_000,
    ) -> None:
        if not modulation_orders:
            raise ValueError("At least one modulation order is required")
        if max_points_per_order < 2:
            raise ValueError("max_points_per_order must be at least 2")

        self.modulation_orders = tuple(modulation_orders)
        self.max_points_per_order = max_points_per_order
        self._figure: Figure | None = None
        self._axes_by_order: dict[int, Axes] = {}
        self._colorbar_created = False
        self._plotted_orders: set[int] = set()

    async def process(
        self,
        stream: Stream[PacketAudioSampleBlock],
    ) -> Stream[PacketAudioSampleBlock]:
        async for block in stream:
            config = block.metadata.get("ask_config")
            if not isinstance(config, ASKConfig):
                raise ValueError(
                    "ASKSignalSpacePlot requires ASKConfig in "
                    "block.metadata['ask_config']"
                )

            self._update_plot(block, config)
            yield block

    def _initialize_plot(self) -> None:
        columns = min(4, len(self.modulation_orders))
        rows = math.ceil(len(self.modulation_orders) / columns)
        self._figure, axes = plt.subplots(
            rows,
            columns,
            figsize=(8.0 * columns, 7.4 * rows),
            constrained_layout=True,
            squeeze=False,
        )
        flat_axes = axes.ravel()

        for order, axis in zip(self.modulation_orders, flat_axes):
            self._axes_by_order[order] = axis
            axis.set_title(f"{order}-ASK (waiting)")
            axis.set_xlabel("Normalized in-phase")
            axis.set_ylabel("Normalized quadrature")
            axis.grid(True, alpha=0.25)

        for axis in flat_axes[len(self.modulation_orders):]:
            axis.set_visible(False)

        self._figure.suptitle(
            "ASK signal space — color progresses from early to late symbols"
        )

    def _update_plot(
        self,
        block: PacketAudioSampleBlock,
        config: ASKConfig,
    ) -> None:
        if self._figure is None:
            self._initialize_plot()
        assert self._figure is not None
        if config.order not in self._axes_by_order:
            raise ValueError(
                f"Unexpected ASK order for signal-space plot: {config.order}"
            )

        coordinates = _ask_symbol_coordinates(block.data, config)
        if len(coordinates) < 2:
            raise ValueError("ASK packet must contain a pilot and payload symbols")

        pilot = coordinates[0]
        if abs(pilot) <= 1e-12:
            raise ValueError("ASK pilot has zero signal-space magnitude")

        # Division by the pilot removes channel gain and common phase. The
        # remaining quadrature movement therefore makes phase drift over the
        # packet visible while the in-phase clusters show amplitude separation.
        payload = coordinates[1:] / pilot
        time_position = np.linspace(0.0, 1.0, len(payload))
        if len(payload) > self.max_points_per_order:
            indices = np.linspace(
                0,
                len(payload) - 1,
                self.max_points_per_order,
                dtype=int,
            )
            payload = payload[indices]
            time_position = time_position[indices]

        axis = self._axes_by_order[config.order]
        axis.clear()
        scatter = axis.scatter(
            payload.real,
            payload.imag,
            c=time_position,
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
            s=15,
            alpha=0.7,
            linewidths=0.0,
            zorder=3,
        )
        axis.scatter(
            config.amplitude_levels,
            np.zeros(config.order),
            marker="x",
            color="black",
            s=24,
            linewidths=0.8,
            label="Ideal levels",
            zorder=1,
        )
        axis.scatter(
            [1.0],
            [0.0],
            marker="*",
            color="tab:red",
            s=55,
            label="Pilot",
            zorder=4,
        )

        plot_limit = max(1.0, float(np.quantile(np.abs(payload), 0.995))) * 1.1
        axis.set_xlim(-plot_limit, plot_limit)
        axis.set_ylim(-plot_limit, plot_limit)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel("Pilot-normalized in-phase")
        axis.set_ylabel("Pilot-normalized quadrature")
        axis.set_title(
            f"{config.order}-ASK — {len(coordinates) - 1} payload symbols"
        )
        axis.grid(True, alpha=0.25)
        axis.legend(loc="upper left", fontsize=7)

        if not self._colorbar_created:
            self._figure.colorbar(
                scatter,
                ax=list(self._axes_by_order.values()),
                label="Position within packet (early → late)",
                shrink=0.8,
            )
            self._colorbar_created = True

        self._plotted_orders.add(config.order)
        self._figure.canvas.draw()

        # Some IDE plot viewers capture a static image on the first show() call.
        # Showing the newly initialized, still-empty grid would therefore leave
        # the user with eight "waiting" panels even though later canvas draws
        # contain data. Display only once every requested order has been plotted.
        if self._plotted_orders == set(self.modulation_orders):
            SIGNAL_SPACE_PLOT_OUTPUT_PATH.parent.mkdir(
                parents=True,
                exist_ok=True,
            )
            self._figure.savefig(
                SIGNAL_SPACE_PLOT_OUTPUT_PATH,
                format="svg",
            )
            print(f"Plot written to {SIGNAL_SPACE_PLOT_OUTPUT_PATH}")
            plt.show(block=False)


@dataclass(frozen=True, slots=True)
class _LocatedPacket:
    run_index: int
    modulation_order: int
    preamble_start: int
    nominal_packet_start: int
    packet_samples: int
    preamble_score: float


class MeasureASKOrderBER(Stage[AudioSampleBlock, ASKOrderBERBlock]):
    """Detect each ASK-order packet and calculate one BER point per order."""

    def __init__(
        self,
        *,
        carrier_frequency_hz: float,
        modulation_orders: tuple[int, ...] = MODULATION_ORDERS,
        symbol_time_seconds: float = SYMBOL_TIME_SECONDS,
        bit_pattern: np.ndarray = BIT_PATTERN,
        preamble: np.ndarray | None = None,
        guard_duration_seconds: float = GUARD_DURATION_SECONDS,
        detection_threshold: float = DETECTION_THRESHOLD,
        sample_rate: int = SAMPLE_RATE,
        max_capture_seconds: float = MEASUREMENT_TIMEOUT_SECONDS,
        signal_space_plot: ASKSignalSpacePlot | None = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.carrier_frequency_hz = carrier_frequency_hz
        self.modulation_orders = tuple(modulation_orders)
        self.symbol_time_seconds = symbol_time_seconds
        self.samples_per_symbol = round(symbol_time_seconds * sample_rate)
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
        if not 0.0 < self.carrier_frequency_hz < self.sample_rate / 2:
            raise ValueError("carrier_frequency_hz must be below Nyquist")
        if self.samples_per_symbol <= 0:
            raise ValueError("symbol_time_seconds rounded to zero samples")
        if not self.modulation_orders:
            raise ValueError("At least one modulation order is required")
        if any(
            later <= earlier
            for earlier, later in zip(
                self.modulation_orders,
                self.modulation_orders[1:],
            )
        ):
            raise ValueError("Modulation orders must strictly increase each run")
        if any(not _is_power_of_two(order) for order in self.modulation_orders):
            raise ValueError("ASK modulation orders must be powers of two")
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

        for order in self.modulation_orders:
            bits_per_symbol = int(math.log2(order))
            if len(self.bit_pattern) % bits_per_symbol:
                raise ValueError(
                    "bit_pattern length must be divisible by bits_per_symbol "
                    f"for order {order}"
                )

    async def process(
        self,
        stream: Stream[AudioSampleBlock],
    ) -> Stream[ASKOrderBERBlock]:
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
            located_packets = self._try_locate_packets()
            if located_packets is None:
                continue

            result = await self._measure_packets(located_packets)
            yield ASKOrderBERBlock(
                data=result,
                is_final=True,
                metadata={"sample_rate": self.sample_rate},
            )
            return

        raise RuntimeError("The audio stream ended before all ASK runs arrived")

    def _trim_buffer(self) -> None:
        if len(self._buffer) <= self.max_capture_samples:
            return
        dropped_samples = len(self._buffer) - self.max_capture_samples
        self._buffer = self._buffer[dropped_samples:]
        self._absolute_offset += dropped_samples
        self._samples_at_last_attempt = max(
            0,
            self._samples_at_last_attempt - dropped_samples,
        )

    def _try_locate_packets(self) -> tuple[_LocatedPacket, ...] | None:
        if len(self._buffer) < len(self.preamble):
            return None

        scores = np.abs(_normalized_correlation(self._buffer, self.preamble))
        shortest_packet_samples = self._packet_samples(
            self.modulation_orders[-1]
        )
        shortest_frame_samples = (
            len(self.preamble)
            + 2 * self.guard_samples
            + shortest_packet_samples
        )
        peaks, _ = find_peaks(
            scores,
            height=self.detection_threshold,
            distance=max(len(self.preamble), shortest_frame_samples // 2),
        )
        if len(peaks) < len(self.modulation_orders):
            return None

        packets: list[_LocatedPacket] = []
        for run_index, (preamble_start, order) in enumerate(
            zip(peaks[:len(self.modulation_orders)], self.modulation_orders),
            start=1,
        ):
            nominal_packet_start = (
                int(preamble_start) + len(self.preamble) + self.guard_samples
            )
            packet_samples = self._packet_samples(order)
            timing_radius = self.samples_per_symbol // 2
            if (
                nominal_packet_start + packet_samples + timing_radius
                > len(self._buffer)
            ):
                return None

            packets.append(
                _LocatedPacket(
                    run_index=run_index,
                    modulation_order=order,
                    preamble_start=int(preamble_start),
                    nominal_packet_start=nominal_packet_start,
                    packet_samples=packet_samples,
                    preamble_score=float(scores[preamble_start]),
                )
            )

        return tuple(packets)

    def _packet_samples(self, order: int) -> int:
        bits_per_symbol = int(math.log2(order))
        payload_symbols = len(self.bit_pattern) // bits_per_symbol
        return (1 + payload_symbols) * self.samples_per_symbol

    async def _measure_packets(
        self,
        packets: tuple[_LocatedPacket, ...],
    ) -> ASKOrderBERResult:
        run_results = []
        for packet in packets:
            run_results.append(await self._measure_packet(packet))

        return ASKOrderBERResult(
            sample_rate=self.sample_rate,
            carrier_frequency_hz=self.carrier_frequency_hz,
            bit_pattern=self.bit_pattern.copy(),
            runs=tuple(run_results),
        )

    async def _measure_packet(
        self,
        packet: _LocatedPacket,
    ) -> ASKOrderRunResult:
        config = _make_ask_config(
            order=packet.modulation_order,
            samples_per_symbol=self.samples_per_symbol,
            carrier_frequency_hz=self.carrier_frequency_hz,
            sample_rate=self.sample_rate,
        )
        timing_offset = self._find_timing_offset(packet, config)
        packet_start = packet.nominal_packet_start + timing_offset
        packet_audio = self._buffer[
            packet_start:packet_start + packet.packet_samples
        ]

        decoder = ASKAudioSamplesToBits(config)
        packet_stream = self._packet_source(
            packet_audio,
            packet,
            packet_start,
            config,
        )
        if self.signal_space_plot is not None:
            packet_stream = self.signal_space_plot.process(packet_stream)
        decoded_stream = decoder.process(packet_stream)
        decoded_block = await anext(decoded_stream)
        estimated_channel_gain = float(decoder.channel_gain or 0.0)
        await decoded_stream.aclose()

        received_bits = np.asarray(decoded_block.data, dtype=np.uint8).reshape(-1)
        comparable_bits = min(len(received_bits), len(self.bit_pattern))
        bit_errors = int(
            np.count_nonzero(
                received_bits[:comparable_bits]
                != self.bit_pattern[:comparable_bits]
            )
            + abs(len(received_bits) - len(self.bit_pattern))
        )
        bit_errors = min(bit_errors, len(self.bit_pattern))
        bits_per_symbol = config.bits_per_symbol
        actual_symbol_time = self.samples_per_symbol / self.sample_rate

        return ASKOrderRunResult(
            run_index=packet.run_index,
            modulation_order=packet.modulation_order,
            bits_per_symbol=bits_per_symbol,
            symbol_time_seconds=actual_symbol_time,
            samples_per_symbol=self.samples_per_symbol,
            compared_bits=len(self.bit_pattern),
            bit_errors=bit_errors,
            ber=bit_errors / len(self.bit_pattern),
            estimated_channel_gain=estimated_channel_gain,
            estimated_level_spacing=(
                estimated_channel_gain
                * config.amplitude
                / (packet.modulation_order - 1)
            ),
            timing_offset_samples=timing_offset,
            preamble_score=packet.preamble_score,
            preamble_start_sample=(
                self._absolute_offset + packet.preamble_start
            ),
        )

    def _find_timing_offset(
        self,
        packet: _LocatedPacket,
        config: ASKConfig,
    ) -> int:
        best_offset = 0
        best_pilot_level = -np.inf
        timing_radius = self.samples_per_symbol // 2

        for timing_offset in range(-timing_radius, timing_radius + 1):
            start = packet.nominal_packet_start + timing_offset
            stop = start + self.samples_per_symbol
            if start < 0 or stop > len(self._buffer):
                continue
            pilot_level = _estimate_carrier_level(
                self._buffer[start:stop],
                config,
            )
            if pilot_level > best_pilot_level:
                best_offset = timing_offset
                best_pilot_level = pilot_level

        return best_offset

    async def _packet_source(
        self,
        packet_audio: np.ndarray,
        packet: _LocatedPacket,
        packet_start: int,
        config: ASKConfig,
    ) -> Stream[PacketAudioSampleBlock]:
        yield PacketAudioSampleBlock(
            data=packet_audio,
            packet_index=packet.run_index - 1,
            preamble_start=self._absolute_offset + packet.preamble_start,
            packet_start=self._absolute_offset + packet_start,
            correlation_score=packet.preamble_score,
            is_final=True,
            metadata={
                "sample_rate": self.sample_rate,
                "modulation_order": packet.modulation_order,
                "ask_config": config,
            },
        )


async def _single_bit_block(bits: np.ndarray) -> Stream[BitBlock]:
    yield BitBlock(data=bits, is_final=True)


async def ask_order_probe_source(
    *,
    carrier_frequency_hz: float = CARRIER_FREQUENCY_HZ,
    modulation_orders: tuple[int, ...] = MODULATION_ORDERS,
    symbol_time_seconds: float = SYMBOL_TIME_SECONDS,
    sample_rate: int = SAMPLE_RATE,
) -> Stream[AudioSampleBlock]:
    """Build the complete preamble/guard/M-ASK order sweep."""

    samples_per_symbol = round(symbol_time_seconds * sample_rate)
    if samples_per_symbol <= 0:
        raise ValueError("symbol_time_seconds rounded to zero samples")

    preamble = create_experiment_preamble(sample_rate)
    guard_samples = round(GUARD_DURATION_SECONDS * sample_rate)
    parts = [
        np.zeros(round(LEADING_SILENCE_SECONDS * sample_rate), dtype=np.float64)
    ]

    for order in modulation_orders:
        config = _make_ask_config(
            order=order,
            samples_per_symbol=samples_per_symbol,
            carrier_frequency_hz=carrier_frequency_hz,
            sample_rate=sample_rate,
        )
        if len(BIT_PATTERN) % config.bits_per_symbol:
            raise ValueError(
                f"BIT_PATTERN is incomplete for order {order} ASK symbols"
            )

        modulated = BitsToASKAudioSamples(config).process(
            _single_bit_block(BIT_PATTERN)
        )
        guarded = AddGuardSilence(guard_samples).process(modulated)
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
            "carrier_frequency_hz": carrier_frequency_hz,
            "symbol_time_seconds": samples_per_symbol / sample_rate,
            "purpose": "ask_ber_modulation_order_sweep",
        },
    )


def write_ask_order_ber_csv(
    result: ASKOrderBERResult,
    path: str | Path = CSV_OUTPUT_PATH,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = np.array(
        [
            (
                run.run_index,
                result.carrier_frequency_hz,
                run.modulation_order,
                run.bits_per_symbol,
                run.symbol_time_seconds,
                run.symbol_rate_baud,
                run.raw_bit_rate_bps,
                run.samples_per_symbol,
                run.compared_bits,
                run.bit_errors,
                run.ber,
                run.estimated_channel_gain,
                run.estimated_level_spacing,
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
            "run_index,carrier_frequency_hz,modulation_order,bits_per_symbol,"
            "symbol_time_seconds,symbol_rate_baud,raw_bit_rate_bps,"
            "samples_per_symbol,compared_bits,bit_errors,ber,"
            "estimated_channel_gain,estimated_level_spacing,"
            "timing_offset_samples,preamble_score,preamble_start_sample"
        ),
        comments="",
        fmt=(
            "%d", "%.8f", "%d", "%d", "%.9f", "%.8f", "%.8f", "%d",
            "%d", "%d", "%.9f", "%.9f", "%.9f", "%d", "%.9f", "%d",
        ),
    )
    print(f"CSV written to {output_path}")


def plot_ask_order_ber(result: ASKOrderBERResult) -> None:
    orders = np.array([run.modulation_order for run in result.runs])
    ber = np.array([run.ber for run in result.runs])
    symbol_time_ms = result.runs[0].symbol_time_seconds * 1e3

    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    ax.plot(orders, ber, marker="o")
    ax.set_xscale("log", base=2)
    ax.set_xticks(orders, labels=[str(order) for order in orders])
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("ASK modulation order")
    ax.set_ylabel("Bit error rate")
    ax.set_title(
        f"ASK BER versus modulation order at "
        f"{result.carrier_frequency_hz:g} Hz and {symbol_time_ms:g} ms/symbol"
    )
    ax.grid(True, which="both", alpha=0.3)
    PLOT_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOT_OUTPUT_PATH, format="svg")
    print(f"Plot written to {PLOT_OUTPUT_PATH}")
    plt.show()


def print_result(result: ASKOrderBERResult) -> None:
    symbol_time_ms = result.runs[0].symbol_time_seconds * 1e3
    print("\nASK modulation-order sweep")
    print(f"  Carrier:       {result.carrier_frequency_hz:g} Hz")
    print(f"  Symbol time:   {symbol_time_ms:g} ms")
    print(f"  Receiver rate: {result.sample_rate} samples/s")
    print(f"  Pattern bits:  {len(result.bit_pattern)} per run")
    print("\n  run   order   bits/sym   raw rate       errors       BER")
    for run in result.runs:
        print(
            f"  {run.run_index:>3}   {run.modulation_order:>5}"
            f"   {run.bits_per_symbol:>8}"
            f"   {run.raw_bit_rate_bps:>8.1f} bps"
            f"   {run.bit_errors:>4}/{run.compared_bits:<4}"
            f"   {run.ber:.6f}"
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
    carrier_frequency_hz: float = CARRIER_FREQUENCY_HZ,
    modulation_orders: tuple[int, ...] = MODULATION_ORDERS,
    symbol_time_seconds: float = SYMBOL_TIME_SECONDS,
    csv_output_path: str | Path = CSV_OUTPUT_PATH,
    show_plot: bool = True,
) -> None:
    if not 0.0 < carrier_frequency_hz < SAMPLE_RATE / 2:
        raise ValueError("Carrier frequency must be between 0 Hz and Nyquist")

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
        ASKSignalSpacePlot(modulation_orders) if show_plot else None
    )
    measurement = MeasureASKOrderBER(
        carrier_frequency_hz=carrier_frequency_hz,
        modulation_orders=modulation_orders,
        symbol_time_seconds=symbol_time_seconds,
        signal_space_plot=signal_space_plot,
    )
    transmit_task = asyncio.create_task(
        sender.consume(
            ask_order_probe_source(
                carrier_frequency_hz=carrier_frequency_hz,
                modulation_orders=modulation_orders,
                symbol_time_seconds=symbol_time_seconds,
            )
        )
    )

    try:
        await _wait_for_microphone(sender)

        async def receive_measurement() -> ASKOrderBERResult:
            async for block in measurement.process(receiver.stream()):
                return block.data
            raise RuntimeError("No ASK order-sweep result was produced")

        result, _ = await asyncio.wait_for(
            asyncio.gather(receive_measurement(), transmit_task),
            timeout=MEASUREMENT_TIMEOUT_SECONDS,
        )
        print_result(result)
        write_ask_order_ber_csv(result, csv_output_path)
        if show_plot:
            plot_ask_order_ber(result)
    finally:
        if not transmit_task.done():
            transmit_task.cancel()
        with suppress(asyncio.CancelledError, Exception):
            await transmit_task
        try:
            await asyncio.to_thread(sender._stop_blocking)
        except Exception:
            pass


def _parse_orders(value: str) -> tuple[int, ...]:
    try:
        orders = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Orders must be comma-separated integers"
        ) from exc
    if not orders or any(not _is_power_of_two(order) for order in orders):
        raise argparse.ArgumentTypeError(
            "Every ASK order must be a power of two and at least 2"
        )
    if any(later <= earlier for earlier, later in zip(orders, orders[1:])):
        raise argparse.ArgumentTypeError("ASK orders must strictly increase")
    if any(BIT_PATTERN_LENGTH % int(math.log2(order)) for order in orders):
        raise argparse.ArgumentTypeError(
            f"Every bits-per-symbol value must divide {BIT_PATTERN_LENGTH}"
        )
    return orders


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure ASK BER while increasing the modulation order."
    )
    parser.add_argument(
        "--frequency-hz",
        type=float,
        default=CARRIER_FREQUENCY_HZ,
        help=f"ASK carrier frequency (default: {CARRIER_FREQUENCY_HZ:g} Hz)",
    )
    parser.add_argument(
        "--symbol-time-ms",
        type=float,
        default=SYMBOL_TIME_SECONDS * 1e3,
        help=f"Fixed symbol time (default: {SYMBOL_TIME_SECONDS * 1e3:g} ms)",
    )
    parser.add_argument(
        "--orders",
        type=_parse_orders,
        default=MODULATION_ORDERS,
        help="Comma-separated ASK orders (default: 2,4,8,16,32,64,128,256)",
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
        help="Disable both the signal-space and BER plots.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = _parse_args()
    asyncio.run(
        main(
            carrier_frequency_hz=arguments.frequency_hz,
            modulation_orders=arguments.orders,
            symbol_time_seconds=arguments.symbol_time_ms / 1e3,
            csv_output_path=arguments.csv,
            show_plot=not arguments.no_plot,
        )
    )
