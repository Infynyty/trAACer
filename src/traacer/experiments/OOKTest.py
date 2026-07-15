from __future__ import annotations

import argparse
import asyncio
import threading
import time
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
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
    BitsToOOKAudioSamples,
    OOKConfig,
)
from traacer.network_stack.layer.physical_layer.synchronization import (
    PrependPreamble,
    create_chirp_preamble,
)
from traacer.receiver.server import cert_path


SAMPLE_RATE = 48_000
CARRIER_FREQUENCY_HZ = 8_000.0
SYMBOL_TIMES_SECONDS = (
    20e-3,
    12e-3,
    8e-3,
    5e-3,
    3e-3,
    2e-3,
    1.5e-3,
    1e-3,
    0.75e-3,
    0.5e-3,
    0.1e-3,
)

# The same deterministic, balanced payload is sent at every symbol time. Repeating
# the 64-bit word gives enough bits for a useful BER estimate without making the
# slowest run excessively long.
BIT_PATTERN_WORD = (
    "11001010011100010110100011110010"
    "00101101100011101001011100010100"
)
BIT_PATTERN_REPETITIONS = 4
BIT_PATTERN = np.tile(
    np.fromiter((int(bit) for bit in BIT_PATTERN_WORD), dtype=np.uint8),
    BIT_PATTERN_REPETITIONS,
)

# These known symbols calibrate the OOK decision threshold and are not included in
# the BER. Alternating values also make symbol-timing refinement unambiguous.
CALIBRATION_BITS = np.tile(np.array([0, 1], dtype=np.uint8), 8)

PREAMBLE_START_FREQUENCY_HZ = 500.0
PREAMBLE_END_FREQUENCY_HZ = 16_000.0
PREAMBLE_DURATION_SECONDS = 0.08
PREAMBLE_AMPLITUDE = 0.8
GUARD_DURATION_SECONDS = 0.08
LEADING_SILENCE_SECONDS = 0.5
TRAILING_SILENCE_SECONDS = 0.5
OOK_AMPLITUDE = 0.8
DETECTION_THRESHOLD = 0.20
MEASUREMENT_TIMEOUT_SECONDS = 120.0
CSV_OUTPUT_PATH = Path("measurements/ask_ook_ber.csv")


@dataclass(frozen=True, slots=True)
class OOKRunResult:
    run_index: int
    symbol_time_seconds: float
    samples_per_symbol: int
    compared_bits: int
    bit_errors: int
    ber: float
    decision_threshold: float
    zero_level: float
    one_level: float
    timing_offset_samples: int
    preamble_score: float
    preamble_start_sample: int

    @property
    def symbol_rate_baud(self) -> float:
        return 1.0 / self.symbol_time_seconds


@dataclass(frozen=True, slots=True)
class OOKBERResult:
    sample_rate: int
    carrier_frequency_hz: float
    bit_pattern: np.ndarray
    runs: tuple[OOKRunResult, ...]


@dataclass(frozen=True, slots=True)
class OOKBERBlock(DataBlock[OOKBERResult]):
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
    # FFT correlation leaves tiny numerical residuals in exact silence. Dividing
    # those by an almost-zero window norm creates false peaks, so silent windows
    # are explicitly kept at score zero.
    normalized = np.zeros_like(correlation, dtype=np.float64)
    non_silent = window_energy > reference_energy * 1e-12
    normalized[non_silent] = correlation[non_silent] / (
        window_energy[non_silent] * reference_energy
    )
    return normalized


def _carrier_levels(
    samples: np.ndarray,
    *,
    samples_per_symbol: int,
    sample_rate: int,
    carrier_frequency_hz: float,
) -> np.ndarray:
    if len(samples) % samples_per_symbol:
        raise ValueError("OOK packet does not contain complete symbols")

    symbols = samples.reshape(-1, samples_per_symbol)
    t = np.arange(samples_per_symbol, dtype=np.float64) / sample_rate
    angular_phase = 2.0 * np.pi * carrier_frequency_hz * t
    sin_template = np.sin(angular_phase)
    cos_template = np.cos(angular_phase)
    sin_energy = float(np.dot(sin_template, sin_template))
    cos_energy = float(np.dot(cos_template, cos_template))

    if sin_energy <= 0.0 or cos_energy <= 0.0:
        raise ValueError("OOK correlation template has zero energy")

    in_phase = symbols @ sin_template / sin_energy
    quadrature = symbols @ cos_template / cos_energy
    return np.hypot(in_phase, quadrature)


class MeasureOOKBER(Stage[AudioSampleBlock, OOKBERBlock]):
    """Detect all test packets and calculate one BER point per symbol time."""

    def __init__(
        self,
        *,
        carrier_frequency_hz: float,
        symbol_times_seconds: tuple[float, ...] = SYMBOL_TIMES_SECONDS,
        bit_pattern: np.ndarray = BIT_PATTERN,
        calibration_bits: np.ndarray = CALIBRATION_BITS,
        preamble: np.ndarray | None = None,
        guard_duration_seconds: float = GUARD_DURATION_SECONDS,
        detection_threshold: float = DETECTION_THRESHOLD,
        sample_rate: int = SAMPLE_RATE,
        max_capture_seconds: float = MEASUREMENT_TIMEOUT_SECONDS,
    ) -> None:
        self.sample_rate = sample_rate
        self.carrier_frequency_hz = carrier_frequency_hz
        self.symbol_times_seconds = tuple(symbol_times_seconds)
        self.bit_pattern = np.asarray(bit_pattern, dtype=np.uint8).reshape(-1)
        self.calibration_bits = np.asarray(
            calibration_bits,
            dtype=np.uint8,
        ).reshape(-1)
        self.preamble = np.asarray(
            create_experiment_preamble(sample_rate)
            if preamble is None
            else preamble,
            dtype=np.float64,
        ).reshape(-1)
        self.guard_samples = round(guard_duration_seconds * sample_rate)
        self.detection_threshold = detection_threshold
        self.max_capture_samples = round(max_capture_seconds * sample_rate)

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
        if not self.symbol_times_seconds:
            raise ValueError("At least one symbol time is required")
        if any(value <= 0.0 for value in self.symbol_times_seconds):
            raise ValueError("All symbol times must be positive")
        if any(
            later >= earlier
            for earlier, later in zip(
                self.symbol_times_seconds,
                self.symbol_times_seconds[1:],
            )
        ):
            raise ValueError("Symbol times must strictly decrease each run")
        if len(self.preamble) == 0:
            raise ValueError("preamble cannot be empty")
        if self.guard_samples < 0:
            raise ValueError("guard_duration_seconds cannot be negative")
        if not 0.0 < self.detection_threshold <= 1.0:
            raise ValueError("detection_threshold must be in (0, 1]")
        for name, bits in (
            ("bit_pattern", self.bit_pattern),
            ("calibration_bits", self.calibration_bits),
        ):
            if len(bits) == 0 or np.any((bits != 0) & (bits != 1)):
                raise ValueError(f"{name} must contain only 0 and 1")
        if not np.any(self.calibration_bits == 0) or not np.any(
            self.calibration_bits == 1
        ):
            raise ValueError("calibration_bits must contain both OOK levels")

    async def process(
        self,
        stream: Stream[AudioSampleBlock],
    ) -> Stream[OOKBERBlock]:
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
            result = self._try_measure()
            if result is not None:
                yield OOKBERBlock(
                    data=result,
                    is_final=True,
                    metadata={"sample_rate": self.sample_rate},
                )
                return

        raise RuntimeError("The audio stream ended before all OOK runs arrived")

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

    def _try_measure(self) -> OOKBERResult | None:
        if len(self._buffer) < len(self.preamble):
            return None

        scores = np.abs(_normalized_correlation(self._buffer, self.preamble))
        shortest_symbol_samples = self._samples_per_symbol(
            self.symbol_times_seconds[-1]
        )
        shortest_packet_samples = (
            len(self.preamble)
            + 2 * self.guard_samples
            + (len(self.calibration_bits) + len(self.bit_pattern))
            * shortest_symbol_samples
        )
        peaks, _ = find_peaks(
            scores,
            height=self.detection_threshold,
            distance=max(len(self.preamble), shortest_packet_samples // 2),
        )
        if len(peaks) < len(self.symbol_times_seconds):
            return None

        selected_peaks = peaks[:len(self.symbol_times_seconds)]
        run_results: list[OOKRunResult] = []

        for run_index, (preamble_start, requested_symbol_time) in enumerate(
            zip(selected_peaks, self.symbol_times_seconds),
            start=1,
        ):
            samples_per_symbol = self._samples_per_symbol(requested_symbol_time)
            payload_start = (
                int(preamble_start) + len(self.preamble) + self.guard_samples
            )
            payload_samples = (
                len(self.calibration_bits) + len(self.bit_pattern)
            ) * samples_per_symbol
            timing_radius = samples_per_symbol // 2

            if payload_start + payload_samples + timing_radius > len(self._buffer):
                return None

            run_results.append(
                self._measure_run(
                    run_index=run_index,
                    preamble_start=int(preamble_start),
                    payload_start=payload_start,
                    samples_per_symbol=samples_per_symbol,
                    preamble_score=float(scores[preamble_start]),
                )
            )

        return OOKBERResult(
            sample_rate=self.sample_rate,
            carrier_frequency_hz=self.carrier_frequency_hz,
            bit_pattern=self.bit_pattern.copy(),
            runs=tuple(run_results),
        )

    def _samples_per_symbol(self, symbol_time_seconds: float) -> int:
        samples = round(symbol_time_seconds * self.sample_rate)
        if samples <= 0:
            raise ValueError("A symbol time rounded to zero samples")
        return samples

    def _measure_run(
        self,
        *,
        run_index: int,
        preamble_start: int,
        payload_start: int,
        samples_per_symbol: int,
        preamble_score: float,
    ) -> OOKRunResult:
        all_bits = np.concatenate((self.calibration_bits, self.bit_pattern))
        num_payload_samples = len(all_bits) * samples_per_symbol
        best: tuple[float, int, np.ndarray, float, float] | None = None

        for timing_offset in range(
            -(samples_per_symbol // 2),
            samples_per_symbol // 2 + 1,
        ):
            start = payload_start + timing_offset
            if start < 0 or start + num_payload_samples > len(self._buffer):
                continue

            levels = _carrier_levels(
                self._buffer[start:start + num_payload_samples],
                samples_per_symbol=samples_per_symbol,
                sample_rate=self.sample_rate,
                carrier_frequency_hz=self.carrier_frequency_hz,
            )
            calibration_levels = levels[:len(self.calibration_bits)]
            zero_levels = calibration_levels[self.calibration_bits == 0]
            one_levels = calibration_levels[self.calibration_bits == 1]
            zero_level = float(np.mean(zero_levels))
            one_level = float(np.mean(one_levels))
            pooled_spread = float(np.std(zero_levels) + np.std(one_levels))
            separation = (one_level - zero_level) / (pooled_spread + 1e-12)

            if best is None or separation > best[0]:
                best = (
                    separation,
                    timing_offset,
                    levels,
                    zero_level,
                    one_level,
                )

        if best is None:
            raise RuntimeError("No complete timing candidate was available")

        _, timing_offset, levels, zero_level, one_level = best
        decision_threshold = 0.5 * (zero_level + one_level)
        received_bits = (
            levels[len(self.calibration_bits):] >= decision_threshold
        ).astype(np.uint8)
        bit_errors = int(np.count_nonzero(received_bits != self.bit_pattern))
        symbol_time_seconds = samples_per_symbol / self.sample_rate

        return OOKRunResult(
            run_index=run_index,
            symbol_time_seconds=symbol_time_seconds,
            samples_per_symbol=samples_per_symbol,
            compared_bits=len(self.bit_pattern),
            bit_errors=bit_errors,
            ber=bit_errors / len(self.bit_pattern),
            decision_threshold=decision_threshold,
            zero_level=zero_level,
            one_level=one_level,
            timing_offset_samples=timing_offset,
            preamble_score=preamble_score,
            preamble_start_sample=self._absolute_offset + preamble_start,
        )


async def _single_bit_block(bits: np.ndarray) -> Stream[BitBlock]:
    yield BitBlock(data=bits, is_final=True)


async def ook_probe_source(
    *,
    carrier_frequency_hz: float = CARRIER_FREQUENCY_HZ,
    symbol_times_seconds: tuple[float, ...] = SYMBOL_TIMES_SECONDS,
    sample_rate: int = SAMPLE_RATE,
) -> Stream[AudioSampleBlock]:
    """Build the complete preamble/guard/OOK sweep as one audio probe."""

    preamble = create_experiment_preamble(sample_rate)
    guard_samples = round(GUARD_DURATION_SECONDS * sample_rate)
    transmitted_bits = np.concatenate((CALIBRATION_BITS, BIT_PATTERN))
    parts = [
        np.zeros(round(LEADING_SILENCE_SECONDS * sample_rate), dtype=np.float64)
    ]

    for symbol_time_seconds in symbol_times_seconds:
        samples_per_symbol = round(symbol_time_seconds * sample_rate)
        config = OOKConfig(
            sample_rate=sample_rate,
            symbol_rate=sample_rate / samples_per_symbol,
            carrier_frequency=carrier_frequency_hz,
            amplitude=OOK_AMPLITUDE,
            ramp_duration=min(2e-3, 0.1 * samples_per_symbol / sample_rate),
            continuous_phase=True,
        )
        modulated = BitsToOOKAudioSamples(config).process(
            _single_bit_block(transmitted_bits)
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
            "purpose": "ook_ber_symbol_time_sweep",
        },
    )


def write_ook_ber_csv(
    result: OOKBERResult,
    path: str | Path = CSV_OUTPUT_PATH,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = np.array(
        [
            (
                run.run_index,
                result.carrier_frequency_hz,
                run.symbol_time_seconds,
                run.symbol_rate_baud,
                run.samples_per_symbol,
                run.compared_bits,
                run.bit_errors,
                run.ber,
                run.decision_threshold,
                run.zero_level,
                run.one_level,
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
            "run_index,carrier_frequency_hz,symbol_time_seconds,"
            "symbol_rate_baud,samples_per_symbol,compared_bits,bit_errors,ber,"
            "decision_threshold,zero_level,one_level,timing_offset_samples,"
            "preamble_score,preamble_start_sample"
        ),
        comments="",
        fmt=(
            "%d", "%.8f", "%.9f", "%.8f", "%d", "%d", "%d", "%.9f",
            "%.9f", "%.9f", "%.9f", "%d", "%.9f", "%d",
        ),
    )
    print(f"CSV written to {output_path}")


def plot_ook_ber(result: OOKBERResult) -> None:
    symbol_times_ms = np.array(
        [run.symbol_time_seconds * 1e3 for run in result.runs]
    )
    ber = np.array([run.ber for run in result.runs])

    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    ax.plot(symbol_times_ms, ber, marker="o")
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Symbol time [ms] (shorter / faster to the right)")
    ax.set_ylabel("Bit error rate")
    ax.set_title(
        f"OOK BER versus symbol time at {result.carrier_frequency_hz:g} Hz"
    )
    ax.grid(True, which="both", alpha=0.3)
    plt.show()


def print_result(result: OOKBERResult) -> None:
    print("\nOOK symbol-time sweep")
    print(f"  Carrier:       {result.carrier_frequency_hz:g} Hz")
    print(f"  Receiver rate: {result.sample_rate} samples/s")
    print(f"  Pattern bits:  {len(result.bit_pattern)} per run")
    print("\n  run   symbol time   rate       errors       BER")
    for run in result.runs:
        print(
            f"  {run.run_index:>3}   {run.symbol_time_seconds * 1e3:>8.3f} ms"
            f"   {run.symbol_rate_baud:>7.1f} Bd"
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
    measurement = MeasureOOKBER(
        carrier_frequency_hz=carrier_frequency_hz,
    )
    transmit_task = asyncio.create_task(
        sender.consume(
            ook_probe_source(carrier_frequency_hz=carrier_frequency_hz)
        )
    )

    try:
        await _wait_for_microphone(sender)

        async def receive_measurement() -> OOKBERResult:
            async for block in measurement.process(receiver.stream()):
                return block.data
            raise RuntimeError("No OOK BER result was produced")

        result, _ = await asyncio.wait_for(
            asyncio.gather(receive_measurement(), transmit_task),
            timeout=MEASUREMENT_TIMEOUT_SECONDS,
        )
        print_result(result)
        write_ook_ber_csv(result, csv_output_path)
        if show_plot:
            plot_ook_ber(result)
    finally:
        if not transmit_task.done():
            transmit_task.cancel()
        with suppress(asyncio.CancelledError, Exception):
            await transmit_task
        try:
            await asyncio.to_thread(sender._stop_blocking)
        except Exception:
            pass


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure OOK BER while progressively shortening symbol time."
    )
    parser.add_argument(
        "--frequency-hz",
        type=float,
        default=CARRIER_FREQUENCY_HZ,
        help=f"OOK carrier frequency (default: {CARRIER_FREQUENCY_HZ:g} Hz)",
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
        help="Save and print the result without opening the BER plot.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = _parse_args()
    asyncio.run(
        main(
            carrier_frequency_hz=arguments.frequency_hz,
            csv_output_path=arguments.csv,
            show_plot=not arguments.no_plot,
        )
    )
