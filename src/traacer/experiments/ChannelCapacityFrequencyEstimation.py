from __future__ import annotations

import asyncio
import math
import threading
import time
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import trapezoid
from scipy.signal import correlate, find_peaks, welch

from traacer.network_stack.layer.base import (
    AudioSampleBlock,
    DataBlock,
    Stage,
    Stream,
    run_webserver,
)
from traacer.network_stack.layer.device_layer.WebserverDeviceLayer import (
    WebserverDeviceLayerReceiverSource,
    WebserverDeviceLayerSenderSink,
)
from traacer.network_stack.layer.physical_layer.synchronization import (
    create_chirp_preamble,
)
from traacer.receiver.server import cert_path


SAMPLE_RATE = 48_000
START_FREQUENCY_HZ = 100.0
END_FREQUENCY_HZ = 22000.0
NOISE_DURATION_SECONDS = 2.0
GUARD_DURATION_SECONDS = 0.25
CHIRP_DURATION_SECONDS = 1.0
TRAILING_SILENCE_SECONDS = 0.25
CHIRP_AMPLITUDE = 0.8
NUM_RUNS = 8
MEASUREMENT_TIMEOUT_SECONDS = 90.0
SPECTRA_PLOT_OUTPUT_PATH = Path("measurements/channel-capacity-spectra.svg")
SNR_PLOT_OUTPUT_PATH = Path("measurements/channel-capacity-snr.svg")
CAPACITY_PLOT_OUTPUT_PATH = Path(
    "measurements/channel-capacity-density.svg"
)


@dataclass(frozen=True, slots=True)
class ChannelCapacityResult:
    sample_rate: int
    start_frequency_hz: float
    end_frequency_hz: float
    bandwidth_hz: float
    frequencies_hz: np.ndarray
    frequency_resolution_hz: float
    noise_psd: np.ndarray
    received_psd: np.ndarray
    signal_psd: np.ndarray
    snr_linear_by_frequency: np.ndarray
    snr_db_by_frequency: np.ndarray
    capacity_density_bps_hz: np.ndarray
    capacity_by_frequency_bps: np.ndarray
    noise_power: float
    signal_power: float
    average_snr_linear: float
    average_snr_db: float
    capacity_bps: float
    chirp_start_sample: int
    detection_score: float


@dataclass(frozen=True, slots=True)
class ChannelCapacityExperimentResult:
    runs: tuple[ChannelCapacityResult, ...]
    frequencies_hz: np.ndarray
    mean_noise_psd: np.ndarray
    noise_psd_standard_deviation: np.ndarray
    mean_signal_psd: np.ndarray
    signal_psd_standard_deviation: np.ndarray
    mean_snr_linear_by_frequency: np.ndarray
    snr_linear_standard_deviation_by_frequency: np.ndarray
    mean_capacity_density_bps_hz: np.ndarray
    capacity_density_standard_deviation_bps_hz: np.ndarray
    capacities_bps: np.ndarray
    mean_capacity_bps: float
    capacity_standard_deviation_bps: float


@dataclass(frozen=True, slots=True)
class ChannelCapacityBlock(DataBlock[ChannelCapacityExperimentResult]):
    pass


def _power_to_db(power: float) -> float:
    return 10.0 * math.log10(power) if power > 0.0 else -math.inf


def _powers_to_db(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    result = np.full(values.shape, -math.inf, dtype=np.float64)
    positive = values > 0.0
    result[positive] = 10.0 * np.log10(values[positive])
    return result


def _power_spectral_density(
    samples: np.ndarray,
    sample_rate: int,
    nperseg: int,
) -> tuple[np.ndarray, np.ndarray]:
    samples = np.asarray(samples, dtype=np.float64).reshape(-1)
    if len(samples) < 2:
        raise ValueError("At least two samples are required to estimate a PSD")

    segment_length = min(nperseg, len(samples))
    frequencies, psd = welch(
        samples,
        fs=sample_rate,
        window="hann",
        nperseg=segment_length,
        noverlap=segment_length // 2,
        detrend="constant",
        scaling="density",
    )
    return frequencies, psd


def estimate_channel_capacity(
    noise_samples: np.ndarray,
    chirp_samples: np.ndarray,
    *,
    sample_rate: int,
    start_frequency_hz: float,
    end_frequency_hz: float,
    psd_segment_length: int = 16_384,
    chirp_start_sample: int = 0,
    detection_score: float = 1.0,
) -> ChannelCapacityResult:
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if not 0.0 <= start_frequency_hz < end_frequency_hz:
        raise ValueError("Expected 0 <= start frequency < end frequency")
    if end_frequency_hz >= sample_rate / 2:
        raise ValueError("end_frequency_hz must be below the Nyquist frequency")
    if psd_segment_length < 2:
        raise ValueError("psd_segment_length must be at least 2")

    common_segment_length = min(
        psd_segment_length,
        len(np.asarray(noise_samples).reshape(-1)),
        len(np.asarray(chirp_samples).reshape(-1)),
    )
    frequencies, noise_psd = _power_spectral_density(
        noise_samples,
        sample_rate,
        common_segment_length,
    )
    received_frequencies, received_psd = _power_spectral_density(
        chirp_samples,
        sample_rate,
        common_segment_length,
    )

    if not np.array_equal(frequencies, received_frequencies):
        raise RuntimeError("Noise and chirp PSD frequency grids do not match")

    in_band = (
        (frequencies >= start_frequency_hz)
        & (frequencies <= end_frequency_hz)
    )
    if np.count_nonzero(in_band) < 2:
        raise ValueError("The measurement band contains fewer than two PSD bins")

    frequencies = frequencies[in_band]
    noise_psd = noise_psd[in_band]
    received_psd = received_psd[in_band]
    signal_psd = np.maximum(received_psd - noise_psd, 0.0)

    snr_linear = np.divide(
        signal_psd,
        noise_psd,
        out=np.zeros_like(signal_psd),
        where=noise_psd > 0.0,
    )
    snr_linear[(noise_psd == 0.0) & (signal_psd > 0.0)] = math.inf
    snr_db = _powers_to_db(snr_linear)

    capacity_density = np.log2(1.0 + snr_linear)
    frequency_widths = np.empty_like(frequencies)
    frequency_widths[0] = (frequencies[1] - frequencies[0]) / 2.0
    frequency_widths[-1] = (frequencies[-1] - frequencies[-2]) / 2.0
    frequency_widths[1:-1] = (frequencies[2:] - frequencies[:-2]) / 2.0
    capacity_by_frequency = capacity_density * frequency_widths
    capacity_bps = float(np.sum(capacity_by_frequency))

    noise_power = float(trapezoid(noise_psd, frequencies))
    signal_power = float(trapezoid(signal_psd, frequencies))
    average_snr_linear = (
        signal_power / noise_power
        if noise_power > 0.0
        else math.inf if signal_power > 0.0
        else 0.0
    )

    return ChannelCapacityResult(
        sample_rate=sample_rate,
        start_frequency_hz=start_frequency_hz,
        end_frequency_hz=end_frequency_hz,
        bandwidth_hz=end_frequency_hz - start_frequency_hz,
        frequencies_hz=frequencies,
        frequency_resolution_hz=float(np.median(np.diff(frequencies))),
        noise_psd=noise_psd,
        received_psd=received_psd,
        signal_psd=signal_psd,
        snr_linear_by_frequency=snr_linear,
        snr_db_by_frequency=snr_db,
        capacity_density_bps_hz=capacity_density,
        capacity_by_frequency_bps=capacity_by_frequency,
        noise_power=noise_power,
        signal_power=signal_power,
        average_snr_linear=average_snr_linear,
        average_snr_db=_power_to_db(average_snr_linear),
        capacity_bps=capacity_bps,
        chirp_start_sample=chirp_start_sample,
        detection_score=detection_score,
    )


def summarize_channel_capacity_runs(
    runs: tuple[ChannelCapacityResult, ...],
) -> ChannelCapacityExperimentResult:
    if len(runs) < 2:
        raise ValueError("At least two runs are required to estimate deviation")

    frequencies_hz = runs[0].frequencies_hz
    for run in runs[1:]:
        if not np.array_equal(run.frequencies_hz, frequencies_hz):
            raise ValueError("All runs must use the same frequency grid")

    noise_psds = np.stack([run.noise_psd for run in runs])
    signal_psds = np.stack([run.signal_psd for run in runs])
    snrs_linear = np.stack(
        [run.snr_linear_by_frequency for run in runs]
    )
    capacity_densities = np.stack(
        [run.capacity_density_bps_hz for run in runs]
    )
    capacities_bps = np.asarray(
        [run.capacity_bps for run in runs],
        dtype=np.float64,
    )

    return ChannelCapacityExperimentResult(
        runs=runs,
        frequencies_hz=frequencies_hz,
        mean_noise_psd=np.mean(noise_psds, axis=0),
        noise_psd_standard_deviation=np.std(
            noise_psds,
            axis=0,
            ddof=1,
        ),
        mean_signal_psd=np.mean(signal_psds, axis=0),
        signal_psd_standard_deviation=np.std(
            signal_psds,
            axis=0,
            ddof=1,
        ),
        mean_snr_linear_by_frequency=np.mean(snrs_linear, axis=0),
        snr_linear_standard_deviation_by_frequency=np.std(
            snrs_linear,
            axis=0,
            ddof=1,
        ),
        mean_capacity_density_bps_hz=np.mean(
            capacity_densities,
            axis=0,
        ),
        capacity_density_standard_deviation_bps_hz=np.std(
            capacity_densities,
            axis=0,
            ddof=1,
        ),
        capacities_bps=capacities_bps,
        mean_capacity_bps=float(np.mean(capacities_bps)),
        capacity_standard_deviation_bps=float(
            np.std(capacities_bps, ddof=1)
        ),
    )


class MeasureChannelCapacity(Stage[AudioSampleBlock, ChannelCapacityBlock]):
    """Estimate channel capacity from repeated chirp/noise measurements."""

    def __init__(
        self,
        *,
        start_frequency_hz: float,
        end_frequency_hz: float,
        chirp_duration_seconds: float,
        noise_duration_seconds: float,
        num_runs: int = NUM_RUNS,
        guard_duration_seconds: float = 0.25,
        detection_threshold: float = 0.08,
        max_capture_seconds: float = MEASUREMENT_TIMEOUT_SECONDS,
        fallback_sample_rate: int = SAMPLE_RATE,
        psd_segment_length: int = 16_384,
    ) -> None:
        if chirp_duration_seconds <= 0.0:
            raise ValueError("chirp_duration_seconds must be positive")
        if noise_duration_seconds <= 0.0:
            raise ValueError("noise_duration_seconds must be positive")
        if num_runs < 2:
            raise ValueError("num_runs must be at least 2")
        if guard_duration_seconds < 0.0:
            raise ValueError("guard_duration_seconds cannot be negative")
        if not 0.0 < detection_threshold <= 1.0:
            raise ValueError("detection_threshold must be in (0, 1]")
        minimum_capture = num_runs * (
            noise_duration_seconds
            + guard_duration_seconds
            + chirp_duration_seconds
            + TRAILING_SILENCE_SECONDS
        )
        if max_capture_seconds < minimum_capture:
            raise ValueError(
                "max_capture_seconds is shorter than the required measurement"
            )

        self.start_frequency_hz = start_frequency_hz
        self.end_frequency_hz = end_frequency_hz
        self.chirp_duration_seconds = chirp_duration_seconds
        self.noise_duration_seconds = noise_duration_seconds
        self.num_runs = num_runs
        self.guard_duration_seconds = guard_duration_seconds
        self.detection_threshold = detection_threshold
        self.max_capture_seconds = max_capture_seconds
        self.fallback_sample_rate = fallback_sample_rate
        self.psd_segment_length = psd_segment_length

        self._buffer = np.empty(0, dtype=np.float64)
        self._absolute_offset = 0
        self._sample_rate: int | None = None
        self._reference = np.empty(0, dtype=np.float64)

    async def process(
        self,
        stream: Stream[AudioSampleBlock],
    ) -> Stream[ChannelCapacityBlock]:
        async for block in stream:
            sample_rate = int(
                block.metadata.get("sample_rate") or self.fallback_sample_rate
            )
            self._set_sample_rate(sample_rate)

            samples = np.asarray(block.data, dtype=np.float64).reshape(-1)
            if len(samples):
                self._buffer = np.concatenate((self._buffer, samples))
                self._trim_buffer()

            measurement = self._try_measure()
            if measurement is not None:
                yield ChannelCapacityBlock(
                    data=measurement,
                    is_final=True,
                    metadata={"sample_rate": sample_rate},
                )
                return

        raise RuntimeError(
            "The audio stream ended before all runs were detected"
        )

    def _set_sample_rate(self, sample_rate: int) -> None:
        if sample_rate <= 0:
            raise ValueError("Received an invalid sample rate")
        if self._sample_rate is not None:
            if sample_rate != self._sample_rate:
                raise ValueError("The input sample rate changed during capture")
            return
        if self.end_frequency_hz >= sample_rate / 2:
            raise ValueError(
                "The chirp end frequency must be below the receiver's Nyquist "
                "frequency"
            )

        self._sample_rate = sample_rate
        self._reference = create_chirp_preamble(
            start_frequency=self.start_frequency_hz,
            end_frequency=self.end_frequency_hz,
            duration_in_sec=self.chirp_duration_seconds,
            sampling_frequency=sample_rate,
        )

    def _trim_buffer(self) -> None:
        assert self._sample_rate is not None
        max_samples = round(self.max_capture_seconds * self._sample_rate)
        if len(self._buffer) > max_samples:
            drop = len(self._buffer) - max_samples
            self._buffer = self._buffer[drop:]
            self._absolute_offset += drop

    def _try_measure(self) -> ChannelCapacityExperimentResult | None:
        if self._sample_rate is None:
            return None

        noise_samples = round(
            self.noise_duration_seconds * self._sample_rate
        )
        guard_samples = round(
            self.guard_duration_seconds * self._sample_rate
        )
        required_prefix = noise_samples + guard_samples
        if len(self._buffer) < required_prefix + len(self._reference):
            return None

        scores = self._normalized_correlation(self._buffer, self._reference)
        absolute_scores = np.abs(scores)
        minimum_distance = round(
            (
                self.noise_duration_seconds
                + self.guard_duration_seconds
                + self.chirp_duration_seconds
            )
            * self._sample_rate
            * 0.8
        )
        peaks, _ = find_peaks(
            absolute_scores,
            height=self.detection_threshold,
            distance=minimum_distance,
        )
        valid_peaks = peaks[
            (peaks >= required_prefix)
            & (peaks + len(self._reference) <= len(self._buffer))
        ]
        if len(valid_peaks) < self.num_runs:
            return None

        runs = []
        for chirp_start in valid_peaks[:self.num_runs]:
            noise_end = chirp_start - guard_samples
            noise_start = noise_end - noise_samples
            chirp_end = chirp_start + len(self._reference)
            runs.append(
                estimate_channel_capacity(
                    self._buffer[noise_start:noise_end],
                    self._buffer[chirp_start:chirp_end],
                    sample_rate=self._sample_rate,
                    start_frequency_hz=self.start_frequency_hz,
                    end_frequency_hz=self.end_frequency_hz,
                    chirp_start_sample=(
                        self._absolute_offset + int(chirp_start)
                    ),
                    psd_segment_length=self.psd_segment_length,
                    detection_score=float(absolute_scores[chirp_start]),
                )
            )

        return summarize_channel_capacity_runs(tuple(runs))

    @staticmethod
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
        return correlation / (window_energy * reference_energy + 1e-15)


async def channel_probe_source(
    num_runs: int = NUM_RUNS,
) -> Stream[AudioSampleBlock]:
    """Emit repeated quiet calibration intervals and finite chirps."""

    if num_runs < 2:
        raise ValueError("num_runs must be at least 2")

    leading_silence = np.zeros(
        round(
            (NOISE_DURATION_SECONDS + GUARD_DURATION_SECONDS) * SAMPLE_RATE
        ),
        dtype=np.float64,
    )
    chirp = CHIRP_AMPLITUDE * create_chirp_preamble(
        start_frequency=START_FREQUENCY_HZ,
        end_frequency=END_FREQUENCY_HZ,
        duration_in_sec=CHIRP_DURATION_SECONDS,
        sampling_frequency=SAMPLE_RATE,
    )
    trailing_silence = np.zeros(
        round(TRAILING_SILENCE_SECONDS * SAMPLE_RATE),
        dtype=np.float64,
    )
    parts = []
    for _ in range(num_runs):
        parts.extend((leading_silence, chirp, trailing_silence))

    yield AudioSampleBlock(
        data=np.concatenate(parts),
        is_final=True,
        metadata={"sample_rate": SAMPLE_RATE, "purpose": "capacity_probe"},
    )


def print_result(result: ChannelCapacityExperimentResult) -> None:
    first_run = result.runs[0]
    mean_snr_db = _powers_to_db(result.mean_snr_linear_by_frequency)
    finite_snr = mean_snr_db[np.isfinite(mean_snr_db)]
    median_snr_db = float(np.median(finite_snr)) if len(finite_snr) else -math.inf
    noise_power = float(np.mean([run.noise_power for run in result.runs]))
    signal_power = float(np.mean([run.signal_power for run in result.runs]))
    average_snr_linear = float(
        np.mean([run.average_snr_linear for run in result.runs])
    )
    detection_scores = np.asarray(
        [run.detection_score for run in result.runs]
    )

    print("\nFrequency-selective channel capacity estimate")
    print(
        f"  Band:                 {first_run.start_frequency_hz:.0f}-"
        f"{first_run.end_frequency_hz:.0f} Hz "
        f"({first_run.bandwidth_hz:.0f} Hz)"
    )
    print(f"  Receiver rate:        {first_run.sample_rate} samples/s")
    print(f"  Runs:                 {len(result.runs)}")
    print(f"  Frequency resolution: {first_run.frequency_resolution_hz:.2f} Hz")
    print(
        f"  Detection scores:     {np.min(detection_scores):.3f}-"
        f"{np.max(detection_scores):.3f}"
    )
    print(f"  Mean noise power:     {_power_to_db(noise_power):.2f} dBFS")
    print(f"  Mean signal power:    {_power_to_db(signal_power):.2f} dBFS")
    print(f"  Mean aggregate SNR:   {_power_to_db(average_snr_linear):.2f} dB")
    print(f"  Median bin SNR:       {median_snr_db:.2f} dB")
    print(
        f"  Capacity:             {result.mean_capacity_bps:,.0f} ± "
        f"{result.capacity_standard_deviation_bps:,.0f} bit/s"
    )


def _plot_power_mean_and_deviation(
    ax: plt.Axes,
    frequencies_hz: np.ndarray,
    mean_power: np.ndarray,
    standard_deviation: np.ndarray,
    *,
    label: str,
) -> None:
    mean_db = _powers_to_db(mean_power)
    upper_db = _powers_to_db(mean_power + standard_deviation)
    lower_db = _powers_to_db(
        np.maximum(mean_power - standard_deviation, 0.0)
    )
    finite_values = np.concatenate(
        (mean_db[np.isfinite(mean_db)], upper_db[np.isfinite(upper_db)])
    )
    plot_floor_db = (
        float(np.min(finite_values)) - 10.0 if len(finite_values) else -300.0
    )
    lower_db = np.where(np.isfinite(lower_db), lower_db, plot_floor_db)

    (line,) = ax.plot(frequencies_hz, mean_db, label=f"Mean {label}")
    ax.fill_between(
        frequencies_hz,
        lower_db,
        upper_db,
        color=line.get_color(),
        alpha=0.2,
        label=f"{label} ±1 standard deviation",
    )


def plot_result(result: ChannelCapacityExperimentResult) -> None:
    frequencies = result.frequencies_hz

    fig, ax = plt.subplots(figsize=(12, 4), constrained_layout=True)
    _plot_power_mean_and_deviation(
        ax,
        frequencies,
        result.mean_signal_psd,
        result.signal_psd_standard_deviation,
        label="signal PSD",
    )
    _plot_power_mean_and_deviation(
        ax,
        frequencies,
        result.mean_noise_psd,
        result.noise_psd_standard_deviation,
        label="noise PSD",
    )
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("PSD [dBFS/Hz]")
    ax.set_title("Received signal and noise spectra")
    ax.grid(True)
    ax.legend()
    SPECTRA_PLOT_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(SPECTRA_PLOT_OUTPUT_PATH, format="svg")
    print(f"Plot written to {SPECTRA_PLOT_OUTPUT_PATH}")
    plt.show()

    fig, ax = plt.subplots(figsize=(12, 4), constrained_layout=True)
    _plot_power_mean_and_deviation(
        ax,
        frequencies,
        result.mean_snr_linear_by_frequency,
        result.snr_linear_standard_deviation_by_frequency,
        label="SNR",
    )
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("SNR [dB]")
    ax.set_title("SNR by frequency")
    ax.grid(True)
    ax.legend()
    fig.savefig(SNR_PLOT_OUTPUT_PATH, format="svg")
    print(f"Plot written to {SNR_PLOT_OUTPUT_PATH}")
    plt.show()

    fig, ax = plt.subplots(figsize=(12, 4), constrained_layout=True)
    capacity_lower = np.maximum(
        result.mean_capacity_density_bps_hz
        - result.capacity_density_standard_deviation_bps_hz,
        0.0,
    )
    capacity_upper = (
        result.mean_capacity_density_bps_hz
        + result.capacity_density_standard_deviation_bps_hz
    )
    ax.plot(
        frequencies,
        result.mean_capacity_density_bps_hz,
        label="Mean capacity density",
    )
    ax.fill_between(
        frequencies,
        capacity_lower,
        capacity_upper,
        alpha=0.25,
        label="±1 standard deviation",
    )
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Capacity density [bit/s/Hz]")
    ax.set_title(
        f"Frequency-selective capacity over {len(result.runs)} runs: "
        f"{result.mean_capacity_bps:,.0f} ± "
        f"{result.capacity_standard_deviation_bps:,.0f} bit/s"
    )
    ax.grid(True)
    ax.legend()
    fig.savefig(CAPACITY_PLOT_OUTPUT_PATH, format="svg")
    print(f"Plot written to {CAPACITY_PLOT_OUTPUT_PATH}")
    plt.show()


async def _wait_for_microphone(
    sender: WebserverDeviceLayerSenderSink,
    timeout_seconds: float = 310.0,
) -> None:
    """Wait until the browser has reported its actual microphone sample rate."""

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


async def main() -> None:
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
    measurement = MeasureChannelCapacity(
        start_frequency_hz=START_FREQUENCY_HZ,
        end_frequency_hz=END_FREQUENCY_HZ,
        chirp_duration_seconds=CHIRP_DURATION_SECONDS,
        noise_duration_seconds=NOISE_DURATION_SECONDS,
        num_runs=NUM_RUNS,
        guard_duration_seconds=GUARD_DURATION_SECONDS,
    )

    transmit_task = asyncio.create_task(sender.consume(channel_probe_source()))
    try:
        # Connecting the download stream only after this point ensures its
        # metadata contains the browser's real sample rate.
        await _wait_for_microphone(sender)

        async def receive_measurement() -> ChannelCapacityExperimentResult:
            async for block in measurement.process(receiver.stream()):
                return block.data
            raise RuntimeError("No channel-capacity measurement was produced")

        result, _ = await asyncio.wait_for(
            asyncio.gather(receive_measurement(), transmit_task),
            timeout=MEASUREMENT_TIMEOUT_SECONDS,
        )
        print_result(result)
        plot_result(result)
    finally:
        if not transmit_task.done():
            transmit_task.cancel()
        with suppress(asyncio.CancelledError, Exception):
            await transmit_task
        try:
            await asyncio.to_thread(sender._stop_blocking)
        except Exception:
            # There may be no active recording session if startup failed.
            pass


if __name__ == "__main__":
    asyncio.run(main())
