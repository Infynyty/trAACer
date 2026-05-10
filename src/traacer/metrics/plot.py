import queue

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as sg


def plot_signal(signal: np.ndarray):
    plt.plot(signal)
    plt.title("Signal in Time Domain")
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    plt.show()


def plot_signal_with_signal_boundaries(signal: np.ndarray, symbol_length, xlim=None):
    if xlim:
        plt.xlim(0, xlim)

    plt.plot(signal)

    for i in range(0, len(signal), int(symbol_length)):
        plt.axvline(x=i, color='red', linestyle='--', linewidth=0.8, alpha=0.5)


    plt.title("Signal in Time Domain")
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    plt.show()

def compare_signals(signal1: np.ndarray, signal2: np.ndarray, symbol_length, xlim=None):
    if xlim:
        plt.xlim(0, xlim)

    plt.plot(signal1, color="blue", alpha=0.5, label="Signal 1")
    plt.plot(signal2, color="red", alpha=0.5, label="Signal 2")

    for i in range(0, max(len(signal1), len(signal2)), symbol_length):
        plt.axvline(x=i, color='red', linestyle='--', linewidth=0.8, alpha=0.5)


    plt.title("Signal in Time Domain")
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    plt.show()

def plot_fft(signal: np.ndarray):
    fs = 44100
    window = np.hanning(len(signal))
    signal_windowed = signal * window
    X = np.fft.rfft(signal_windowed)
    freqs = np.fft.rfftfreq(len(signal), 1/fs)
    magnitude = np.abs(X) / len(signal)
    plt.plot(freqs, magnitude)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.title("FFT with Hann Window")
    plt.xlim(0, 20000)
    plt.show()

def plot_spectrogram(signal: np.ndarray, frequency):
    f, t, Sxx = sg.spectrogram(signal, fs=44100)
    plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [s]')
    plt.colorbar(label='Power [dB]')
    plt.show()

q = queue.Queue()

def wait_for_signals():
    signal1 = q.get()
    signal2 = q.get()

    compare_signals(signal1, signal2, 8200)

def plot_constellation(symbols, title):
    plt.figure()
    plt.scatter(np.real(symbols), np.imag(symbols), s=10)
    plt.axhline(0)
    plt.axvline(0)
    plt.grid(True)
    plt.title(title)
    plt.xlabel("Real")
    plt.ylabel("Imag")
    plt.axis("equal")
    plt.show()
