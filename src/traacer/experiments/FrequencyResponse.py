from __future__ import annotations

import asyncio
import threading
import time
from contextlib import suppress
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import correlate, find_peaks, stft

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
END_FREQUENCY_HZ = 22_000.0
CHIRP_DURATION_SECONDS = 2.0
LEADING_SILENCE_SECONDS = 0.5
INTER_CHIRP_SILENCE_SECONDS = 0.5
TRAILING_SILENCE_SECONDS = 0.5
CHIRP_AMPLITUDE = 0.8
NUM_MEASUREMENTS = 8
DETECTION_THRESHOLD = 0.08
STFT_WINDOW_SAMPLES = 2048
STFT_OVERLAP_SAMPLES = 1536
NUM_RESPONSE_POINTS = 500
MEASUREMENT_TIMEOUT_SECONDS = 60.0

CSV_OUTPUT_PATH = "measurements/frequency_response.csv"

def write_frequency_response_csv(
    result: FrequencyResponseResult,
    path: str,
) -> None:
    columns = [
        result.frequencies_hz,
        result.mean_response_db,
        result.standard_deviation_db,
        result.variance_db2,
    ]

    header = [
        "frequency_hz",
        "mean_response_db",
        "standard_deviation_db",
        "variance_db2",
    ]

    for measurement_index, response_db in enumerate(result.responses_db, start=1):
        columns.append(response_db)
        header.append(f"response_{measurement_index}_db")

    data = np.column_stack(columns)

    np.savetxt(
        path,
        data,
        delimiter=",",
        header=",".join(header),
        comments="",
        fmt="%.8f",
    )

    print(f"CSV written to {path}")


@dataclass(frozen=True, slots=True)
class FrequencyResponseResult:
    sample_rate: int
    frequencies_hz: np.ndarray
    responses_db: np.ndarray
    mean_response_db: np.ndarray
    variance_db2: np.ndarray
    standard_deviation_db: np.ndarray
    detection_scores: np.ndarray
    chirp_start_samples: np.ndarray


@dataclass(frozen=True, slots=True)
class FrequencyResponseBlock(DataBlock[FrequencyResponseResult]):
    pass


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


def _extract_chirp_response_db(
    received_chirp: np.ndarray,
    reference_chirp: np.ndarray,
    *,
    sample_rate: int,
    start_frequency_hz: float,
    end_frequency_hz: float,
    num_response_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    nperseg = min(STFT_WINDOW_SAMPLES, len(reference_chirp))
    noverlap = min(STFT_OVERLAP_SAMPLES, nperseg - 1)

    frequencies, times, reference_stft = stft(
        reference_chirp,
        fs=sample_rate,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        boundary=None,
        padded=False,
    )
    _, _, received_stft = stft(
        received_chirp,
        fs=sample_rate,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        boundary=None,
        padded=False,
    )

    instantaneous_frequencies = (
        start_frequency_hz
        + (end_frequency_hz - start_frequency_hz)
        * times
        / (len(reference_chirp) / sample_rate)
    )
    instantaneous_frequencies = np.clip(
        instantaneous_frequencies,
        start_frequency_hz,
        end_frequency_hz,
    )

    ridge_bins = np.abs(
        frequencies[:, None] - instantaneous_frequencies[None, :]
    ).argmin(axis=0)
    time_indices = np.arange(len(times))

    reference_magnitude = np.abs(reference_stft[ridge_bins, time_indices])
    received_magnitude = np.abs(received_stft[ridge_bins, time_indices])

    response_db = 20.0 * np.log10(
        (received_magnitude + 1e-12) / (reference_magnitude + 1e-12)
    )

    target_frequencies = np.linspace(
        start_frequency_hz,
        end_frequency_hz,
        num_response_points,
    )
    interpolated_response = np.interp(
        target_frequencies,
        instantaneous_frequencies,
        response_db,
    )

    return target_frequencies, interpolated_response


class MeasureFrequencyResponse(
    Stage[AudioSampleBlock, FrequencyResponseBlock]
):
    def __init__(
        self,
        *,
        start_frequency_hz: float,
        end_frequency_hz: float,
        chirp_duration_seconds: float,
        num_measurements: int,
        inter_chirp_silence_seconds: float,
        detection_threshold: float = DETECTION_THRESHOLD,
        num_response_points: int = NUM_RESPONSE_POINTS,
        max_capture_seconds: float = 60.0,
        fallback_sample_rate: int = SAMPLE_RATE,
    ) -> None:
        if chirp_duration_seconds <= 0.0:
            raise ValueError("chirp_duration_seconds must be positive")
        if num_measurements < 2:
            raise ValueError("num_measurements must be at least 2")
        if inter_chirp_silence_seconds < 0.0:
            raise ValueError("inter_chirp_silence_seconds cannot be negative")
        if not 0.0 < detection_threshold <= 1.0:
            raise ValueError("detection_threshold must be in (0, 1]")
        if num_response_points < 2:
            raise ValueError("num_response_points must be at least 2")

        self.start_frequency_hz = start_frequency_hz
        self.end_frequency_hz = end_frequency_hz
        self.chirp_duration_seconds = chirp_duration_seconds
        self.num_measurements = num_measurements
        self.inter_chirp_silence_seconds = inter_chirp_silence_seconds
        self.detection_threshold = detection_threshold
        self.num_response_points = num_response_points
        self.max_capture_seconds = max_capture_seconds
        self.fallback_sample_rate = fallback_sample_rate

        self._buffer = np.empty(0, dtype=np.float64)
        self._absolute_offset = 0
        self._sample_rate: int | None = None
        self._reference = np.empty(0, dtype=np.float64)

    async def process(
        self,
        stream: Stream[AudioSampleBlock],
    ) -> Stream[FrequencyResponseBlock]:
        async for block in stream:
            sample_rate = int(
                block.metadata.get("sample_rate") or self.fallback_sample_rate
            )
            self._set_sample_rate(sample_rate)

            samples = np.asarray(block.data, dtype=np.float64).reshape(-1)
            if len(samples):
                self._buffer = np.concatenate((self._buffer, samples))
                self._trim_buffer()

            result = self._try_measure()
            if result is not None:
                yield FrequencyResponseBlock(
                    data=result,
                    is_final=True,
                    metadata={"sample_rate": sample_rate},
                )
                return

        raise RuntimeError(
            "The audio stream ended before all chirps were detected"
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
                "The chirp end frequency must be below the Nyquist frequency"
            )

        self._sample_rate = sample_rate
        self._reference = CHIRP_AMPLITUDE * create_chirp_preamble(
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

    def _try_measure(self) -> FrequencyResponseResult | None:
        if self._sample_rate is None:
            return None
        if len(self._buffer) < len(self._reference):
            return None

        scores = np.abs(
            _normalized_correlation(self._buffer, self._reference)
        )
        minimum_distance = round(
            (
                self.chirp_duration_seconds
                + 0.5 * self.inter_chirp_silence_seconds
            )
            * self._sample_rate
        )
        peaks, properties = find_peaks(
            scores,
            height=self.detection_threshold,
            distance=minimum_distance,
        )

        valid_peaks = peaks[
            peaks + len(self._reference) <= len(self._buffer)
        ]
        if len(valid_peaks) < self.num_measurements:
            return None

        selected_peaks = valid_peaks[:self.num_measurements]
        detection_scores = scores[selected_peaks]

        responses = []
        frequencies_hz: np.ndarray | None = None

        for chirp_start in selected_peaks:
            chirp_end = chirp_start + len(self._reference)
            received_chirp = self._buffer[chirp_start:chirp_end]
            frequencies_hz, response_db = _extract_chirp_response_db(
                received_chirp,
                self._reference,
                sample_rate=self._sample_rate,
                start_frequency_hz=self.start_frequency_hz,
                end_frequency_hz=self.end_frequency_hz,
                num_response_points=self.num_response_points,
            )
            responses.append(response_db)

        responses_db = np.stack(responses)
        mean_response_db = np.mean(responses_db, axis=0)
        variance_db2 = np.var(responses_db, axis=0, ddof=1)
        standard_deviation_db = np.sqrt(variance_db2)

        assert frequencies_hz is not None
        return FrequencyResponseResult(
            sample_rate=self._sample_rate,
            frequencies_hz=frequencies_hz,
            responses_db=responses_db,
            mean_response_db=mean_response_db,
            variance_db2=variance_db2,
            standard_deviation_db=standard_deviation_db,
            detection_scores=detection_scores,
            chirp_start_samples=(
                self._absolute_offset + selected_peaks
            ).astype(np.int64),
        )


async def frequency_response_probe_source() -> Stream[AudioSampleBlock]:
    leading_silence = np.zeros(
        round(LEADING_SILENCE_SECONDS * SAMPLE_RATE),
        dtype=np.float64,
    )
    chirp = CHIRP_AMPLITUDE * create_chirp_preamble(
        start_frequency=START_FREQUENCY_HZ,
        end_frequency=END_FREQUENCY_HZ,
        duration_in_sec=CHIRP_DURATION_SECONDS,
        sampling_frequency=SAMPLE_RATE,
    )
    inter_chirp_silence = np.zeros(
        round(INTER_CHIRP_SILENCE_SECONDS * SAMPLE_RATE),
        dtype=np.float64,
    )
    trailing_silence = np.zeros(
        round(TRAILING_SILENCE_SECONDS * SAMPLE_RATE),
        dtype=np.float64,
    )

    parts = [leading_silence]
    for measurement_index in range(NUM_MEASUREMENTS):
        parts.append(chirp)
        if measurement_index < NUM_MEASUREMENTS - 1:
            parts.append(inter_chirp_silence)
    parts.append(trailing_silence)

    yield AudioSampleBlock(
        data=np.concatenate(parts),
        is_final=True,
        metadata={
            "sample_rate": SAMPLE_RATE,
            "purpose": "frequency_response_probe",
        },
    )


def plot_frequency_response(result: FrequencyResponseResult) -> None:
    lower = result.mean_response_db - result.standard_deviation_db
    upper = result.mean_response_db + result.standard_deviation_db

    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    ax.plot(
        result.frequencies_hz,
        result.mean_response_db,
        label="Mean response",
    )
    ax.fill_between(
        result.frequencies_hz,
        lower,
        upper,
        alpha=0.25,
        label="±1 standard deviation",
    )
    ax.axhline(0.0, linewidth=1.0, linestyle="--")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Relative gain [dB]")
    ax.set_title(
        f"Device frequency response over {len(result.responses_db)} chirps"
    )
    ax.grid(True)
    ax.legend()
    plt.show()


def print_result(result: FrequencyResponseResult) -> None:
    print("\nFrequency-response measurement")
    print(
        f"  Band:             {result.frequencies_hz[0]:.0f}-"
        f"{result.frequencies_hz[-1]:.0f} Hz"
    )
    print(f"  Receiver rate:    {result.sample_rate} samples/s")
    print(f"  Measurements:     {len(result.responses_db)}")
    print(
        f"  Detection scores: "
        f"{np.min(result.detection_scores):.3f}-"
        f"{np.max(result.detection_scores):.3f}"
    )
    print(
        f"  Mean deviation:   "
        f"{np.mean(result.standard_deviation_db):.2f} dB"
    )
    print(
        f"  Maximum deviation: "
        f"{np.max(result.standard_deviation_db):.2f} dB"
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
    measurement = MeasureFrequencyResponse(
        start_frequency_hz=START_FREQUENCY_HZ,
        end_frequency_hz=END_FREQUENCY_HZ,
        chirp_duration_seconds=CHIRP_DURATION_SECONDS,
        num_measurements=NUM_MEASUREMENTS,
        inter_chirp_silence_seconds=INTER_CHIRP_SILENCE_SECONDS,
        detection_threshold=DETECTION_THRESHOLD,
    )

    transmit_task = asyncio.create_task(
        sender.consume(frequency_response_probe_source())
    )

    try:
        await _wait_for_microphone(sender)

        async def receive_measurement() -> FrequencyResponseResult:
            async for block in measurement.process(receiver.stream()):
                return block.data
            raise RuntimeError("No frequency-response result was produced")

        result, _ = await asyncio.wait_for(
            asyncio.gather(receive_measurement(), transmit_task),
            timeout=MEASUREMENT_TIMEOUT_SECONDS,
        )
        print_result(result)
        plot_frequency_response(result)
        write_frequency_response_csv(result, CSV_OUTPUT_PATH)
    finally:
        if not transmit_task.done():
            transmit_task.cancel()
        with suppress(asyncio.CancelledError, Exception):
            await transmit_task
        try:
            await asyncio.to_thread(sender._stop_blocking)
        except Exception:
            pass


if __name__ == "__main__":
    asyncio.run(main())
