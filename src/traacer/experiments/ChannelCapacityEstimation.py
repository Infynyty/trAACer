from __future__ import annotations

import asyncio
import math
import threading
import time
from contextlib import suppress
from dataclasses import dataclass

import numpy as np
from scipy.integrate import trapezoid
from scipy.signal import correlate, welch

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
END_FREQUENCY_HZ = 20_000.0
NOISE_DURATION_SECONDS = 2.0
GUARD_DURATION_SECONDS = 0.25
CHIRP_DURATION_SECONDS = 1.0
TRAILING_SILENCE_SECONDS = 0.25
CHIRP_AMPLITUDE = 0.8
MEASUREMENT_TIMEOUT_SECONDS = 20.0


@dataclass(frozen=True, slots=True)
class ChannelCapacityResult:
    """Measured powers and the resulting Shannon-Hartley estimate.

    Powers are mean-square values relative to full-scale digital audio.  The
    noise floor is also reported as a power spectral density in dBFS/Hz.
    """

    sample_rate: int
    start_frequency_hz: float
    end_frequency_hz: float
    bandwidth_hz: float
    noise_power: float
    noise_floor_dbfs_hz: float
    received_chirp_power: float
    signal_power: float
    signal_strength_dbfs: float
    snr_linear: float
    snr_db: float
    capacity_bps: float
    chirp_start_sample: int
    detection_score: float


@dataclass(frozen=True, slots=True)
class ChannelCapacityBlock(DataBlock[ChannelCapacityResult]):
    pass


def _power_to_db(power: float) -> float:
    return 10.0 * math.log10(power) if power > 0.0 else -math.inf


def _band_power(
    samples: np.ndarray,
    sample_rate: int,
    start_frequency_hz: float,
    end_frequency_hz: float,
) -> float:
    """Return mean-square power within a frequency band using Welch's PSD."""

    samples = np.asarray(samples, dtype=np.float64).reshape(-1)
    if len(samples) < 2:
        raise ValueError("At least two samples are required to estimate power")

    nperseg = min(4096, len(samples))
    frequencies, psd = welch(
        samples,
        fs=sample_rate,
        window="hann",
        nperseg=nperseg,
        noverlap=nperseg // 2,
        detrend="constant",
        scaling="density",
    )
    in_band = (
        (frequencies >= start_frequency_hz)
        & (frequencies <= end_frequency_hz)
    )
    if np.count_nonzero(in_band) < 2:
        raise ValueError("The measurement band contains fewer than two FFT bins")

    return float(trapezoid(psd[in_band], frequencies[in_band]))


def estimate_channel_capacity(
    noise_samples: np.ndarray,
    chirp_samples: np.ndarray,
    *,
    sample_rate: int,
    start_frequency_hz: float,
    end_frequency_hz: float,
    chirp_start_sample: int = 0,
    detection_score: float = 1.0,
) -> ChannelCapacityResult:
    """Estimate in-band signal/noise power and Shannon-Hartley capacity."""

    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if not 0.0 <= start_frequency_hz < end_frequency_hz:
        raise ValueError("Expected 0 <= start frequency < end frequency")
    if end_frequency_hz >= sample_rate / 2:
        raise ValueError("end_frequency_hz must be below the Nyquist frequency")

    bandwidth_hz = end_frequency_hz - start_frequency_hz
    noise_power = _band_power(
        noise_samples,
        sample_rate,
        start_frequency_hz,
        end_frequency_hz,
    )
    received_chirp_power = _band_power(
        chirp_samples,
        sample_rate,
        start_frequency_hz,
        end_frequency_hz,
    )

    # The chirp interval contains signal and noise.  Subtract the independently
    # measured noise contribution before forming S/N.
    signal_power = max(0.0, received_chirp_power - noise_power)
    if noise_power > 0.0:
        snr_linear = signal_power / noise_power
    elif signal_power > 0.0:
        snr_linear = math.inf
    else:
        snr_linear = 0.0

    snr_db = _power_to_db(snr_linear)
    capacity_bps = bandwidth_hz * math.log2(1.0 + snr_linear)

    return ChannelCapacityResult(
        sample_rate=sample_rate,
        start_frequency_hz=start_frequency_hz,
        end_frequency_hz=end_frequency_hz,
        bandwidth_hz=bandwidth_hz,
        noise_power=noise_power,
        noise_floor_dbfs_hz=_power_to_db(noise_power / bandwidth_hz),
        received_chirp_power=received_chirp_power,
        signal_power=signal_power,
        signal_strength_dbfs=_power_to_db(signal_power),
        snr_linear=snr_linear,
        snr_db=snr_db,
        capacity_bps=capacity_bps,
        chirp_start_sample=chirp_start_sample,
        detection_score=detection_score,
    )


class MeasureChannelCapacity(Stage[AudioSampleBlock, ChannelCapacityBlock]):
    """Find a received chirp and estimate capacity from it and preceding noise."""

    def __init__(
        self,
        *,
        start_frequency_hz: float,
        end_frequency_hz: float,
        chirp_duration_seconds: float,
        noise_duration_seconds: float,
        guard_duration_seconds: float = 0.25,
        detection_threshold: float = 0.08,
        max_capture_seconds: float = 15.0,
        fallback_sample_rate: int = SAMPLE_RATE,
    ) -> None:
        if chirp_duration_seconds <= 0.0:
            raise ValueError("chirp_duration_seconds must be positive")
        if noise_duration_seconds <= 0.0:
            raise ValueError("noise_duration_seconds must be positive")
        if guard_duration_seconds < 0.0:
            raise ValueError("guard_duration_seconds cannot be negative")
        if not 0.0 < detection_threshold <= 1.0:
            raise ValueError("detection_threshold must be in (0, 1]")
        minimum_capture = (
            noise_duration_seconds
            + guard_duration_seconds
            + chirp_duration_seconds
        )
        if max_capture_seconds < minimum_capture:
            raise ValueError(
                "max_capture_seconds is shorter than the required measurement"
            )

        self.start_frequency_hz = start_frequency_hz
        self.end_frequency_hz = end_frequency_hz
        self.chirp_duration_seconds = chirp_duration_seconds
        self.noise_duration_seconds = noise_duration_seconds
        self.guard_duration_seconds = guard_duration_seconds
        self.detection_threshold = detection_threshold
        self.max_capture_seconds = max_capture_seconds
        self.fallback_sample_rate = fallback_sample_rate

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
            "The audio stream ended before a complete chirp was detected"
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

    def _try_measure(self) -> ChannelCapacityResult | None:
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
        eligible_scores = np.abs(scores[required_prefix:])
        if len(eligible_scores) == 0:
            return None

        relative_start = int(np.argmax(eligible_scores))
        chirp_start = required_prefix + relative_start
        detection_score = float(eligible_scores[relative_start])
        if detection_score < self.detection_threshold:
            return None

        noise_end = chirp_start - guard_samples
        noise_start = noise_end - noise_samples
        chirp_end = chirp_start + len(self._reference)
        if chirp_end > len(self._buffer):
            return None

        return estimate_channel_capacity(
            self._buffer[noise_start:noise_end],
            self._buffer[chirp_start:chirp_end],
            sample_rate=self._sample_rate,
            start_frequency_hz=self.start_frequency_hz,
            end_frequency_hz=self.end_frequency_hz,
            chirp_start_sample=self._absolute_offset + chirp_start,
            detection_score=detection_score,
        )

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


async def channel_probe_source() -> Stream[AudioSampleBlock]:
    """Emit a quiet calibration interval followed by one finite chirp."""

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
    yield AudioSampleBlock(
        data=np.concatenate((leading_silence, chirp, trailing_silence)),
        is_final=True,
        metadata={"sample_rate": SAMPLE_RATE, "purpose": "capacity_probe"},
    )


def print_result(result: ChannelCapacityResult) -> None:
    print("\nChannel capacity estimate")
    print(
        f"  Band:             {result.start_frequency_hz:.0f}-"
        f"{result.end_frequency_hz:.0f} Hz "
        f"({result.bandwidth_hz:.0f} Hz)"
    )
    print(f"  Receiver rate:    {result.sample_rate} samples/s")
    print(f"  Detection score:  {result.detection_score:.3f}")
    print(f"  Noise floor:      {result.noise_floor_dbfs_hz:.2f} dBFS/Hz")
    print(f"  Noise power:      {_power_to_db(result.noise_power):.2f} dBFS")
    print(f"  Signal strength:  {result.signal_strength_dbfs:.2f} dBFS")
    print(f"  SNR:              {result.snr_db:.2f} dB")
    print(f"  Capacity:         {result.capacity_bps:,.0f} bit/s")


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
        guard_duration_seconds=GUARD_DURATION_SECONDS,
    )

    transmit_task = asyncio.create_task(sender.consume(channel_probe_source()))
    try:
        # Connecting the download stream only after this point ensures its
        # metadata contains the browser's real sample rate.
        await _wait_for_microphone(sender)

        async def receive_measurement() -> ChannelCapacityResult:
            async for block in measurement.process(receiver.stream()):
                return block.data
            raise RuntimeError("No channel-capacity measurement was produced")

        result, _ = await asyncio.wait_for(
            asyncio.gather(receive_measurement(), transmit_task),
            timeout=MEASUREMENT_TIMEOUT_SECONDS,
        )
        print_result(result)
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
