import numpy as np

def create_chirp_preamble(
                       start_frequency=2000,
                       end_frequency=8000,
                       duration_in_sec=0.02,
                       sampling_frequency=44100) -> np.ndarray:

    N = int(duration_in_sec * sampling_frequency)
    t = np.arange(N) / sampling_frequency

    k = (end_frequency - start_frequency) / duration_in_sec

    chirp = np.cos(2 * np.pi * (start_frequency * t + 0.5 * k * t**2))

    window = np.hanning(N)
    return chirp * window

def create_zadoff_chu_preamble() -> np.ndarray:
    N = 839
    u = 25
    t = np.arange(N, dtype=np.float64)

    zadoff_chu = np.exp(-1j * np.pi * u * t * (t + 1) / N)
    window = np.hanning(N).astype(np.float64)

    signal = zadoff_chu * window
    return signal.astype(np.float32)

def find_preamble_start(received_signal: np.ndarray,
                        preamble: np.ndarray) -> int:
    correlation = np.correlate(received_signal, preamble, mode="valid")
    start_index = int(np.argmax(np.abs(correlation)))
    return start_index
