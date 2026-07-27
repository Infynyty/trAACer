from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path

import matplotlib.pyplot as plt
from fontTools.ttLib.woff2 import bboxFormat
from matplotlib.patches import Arc


def plot_waveforms(
    t: np.ndarray,
    signals: Sequence[np.ndarray],
    sample_rate: int,
    labels: Sequence[str] | None = None,
    zoom_time: float | None = 0.01,
    title: str = "Waveforms",
):
    """
    Plot one or more waveforms on the same axes.

    Parameters
    ----------
    t:
        Time axis for all signals.
    signals:
        Sequence of 1D arrays to plot.
    sample_rate:
        Sampling rate in Hz.
    labels:
        Optional labels for legend. Must match number of signals if provided.
    zoom_time:
        If given, only plot the first `zoom_time` seconds.
        If None, plot the full signal.
    title:
        Plot title.
    """
    if len(signals) == 0:
        raise ValueError("signals must contain at least one waveform")

    signal_lengths = {len(s) for s in signals}
    if len(signal_lengths) != 1:
        raise ValueError("all signals must have the same length")

    if len(t) != len(signals[0]):
        raise ValueError("time axis length must match signal length")

    if labels is not None and len(labels) != len(signals):
        raise ValueError("labels length must match signals length")

    if zoom_time is None:
        n = len(t)
    else:
        n = min(len(t), int(sample_rate * zoom_time))

    plt.figure(figsize=(10, 4))

    for i, signal in enumerate(signals):
        label = labels[i] if labels is not None else None
        if i is len(signals) - 1:
            label = "Mixed Signal"
        plt.plot(t[:n], signal[:n], label=label)

    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.title(title)
    plt.grid(True)

    plt.legend()

    plt.tight_layout()
    plt.show()

def _prepare_ask_amplitudes(
    amplitudes: list[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    amplitudes = np.asarray(amplitudes, dtype=float)

    if amplitudes.ndim != 1 or len(amplitudes) < 2:
        raise ValueError(
            "amplitudes must be a one-dimensional array with at least two values."
        )

    if len(np.unique(amplitudes)) != len(amplitudes):
        raise ValueError("amplitudes must not contain duplicate values.")

    sorted_indices = np.argsort(amplitudes)
    sorted_amplitudes = amplitudes[sorted_indices]

    return sorted_amplitudes, sorted_indices


def _configure_ask_axis(
    ax: plt.Axes,
    amplitudes: np.ndarray,
) -> None:
    amplitude_range = amplitudes[-1] - amplitudes[0]
    padding = max(0.1, 0.15 * amplitude_range)

    ax.set_xlim(
        amplitudes[0] - padding,
        amplitudes[-1] + padding,
    )
    ax.set_ylim(-0.5, 0.45)

    ax.axhline(0, linewidth=1)

    ax.set_xlabel("Normalized amplitude", labelpad=2)
    ax.set_yticks([])

    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_position(("data", 0))

    ax.grid(True, axis="x", linestyle="--", alpha=0.5)


def plot_ask_signal_space(
    amplitudes: list[float] | np.ndarray,
    output_path: str | None = None,
) -> None:
    sorted_amplitudes, sorted_indices = _prepare_ask_amplitudes(amplitudes)

    if np.any(sorted_amplitudes < 0):
        raise ValueError(
            "ASK amplitudes must be non-negative. "
            "Negative coefficients represent a phase reversal."
        )

    fig, ax = plt.subplots(
        figsize=(8, 2.4),
        constrained_layout=True,
    )

    ax.scatter(
        sorted_amplitudes,
        np.zeros_like(sorted_amplitudes),
        s=100,
        zorder=3,
    )

    for amplitude, symbol_index in zip(sorted_amplitudes, sorted_indices):
        ax.annotate(
            rf"$m_{{{symbol_index}}}$",
            xy=(amplitude, 0),
            xytext=(0, 14),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=12,
        )

    _configure_ask_axis(ax, sorted_amplitudes)

    ax.set_xticks(sorted_amplitudes)
    ax.set_title(f"{len(sorted_amplitudes)}-ary ASK Signal-Space Diagram")

    if output_path is not None:
        fig.savefig(
            output_path,
            format="svg",
            bbox_inches="tight",
        )

    plt.show()


def plot_ask_signal_space_decision_boundaries(
    amplitudes: list[float] | np.ndarray,
    output_path: str | None = None,
) -> None:
    sorted_amplitudes, sorted_indices = _prepare_ask_amplitudes(amplitudes)

    if np.any(sorted_amplitudes < 0):
        raise ValueError(
            "ASK amplitudes must be non-negative. "
            "Negative coefficients represent a phase reversal."
        )

    decision_boundaries = (
        sorted_amplitudes[:-1] + sorted_amplitudes[1:]
    ) / 2

    fig, ax = plt.subplots(
        figsize=(8, 2.8),
        constrained_layout=True,
    )

    ax.scatter(
        sorted_amplitudes,
        np.zeros_like(sorted_amplitudes),
        s=100,
        zorder=3,
    )

    for amplitude, symbol_index in zip(sorted_amplitudes, sorted_indices):
        ax.annotate(
            rf"$m_{{{symbol_index}}}$",
            xy=(amplitude, 0),
            xytext=(0, 14),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=12,
        )

    for boundary_index, boundary in enumerate(decision_boundaries):
        ax.axvline(
            boundary,
            color="red",
            linestyle="--",
            linewidth=1.5,
            zorder=2,
        )

        ax.annotate(
            rf"$\gamma_{{{boundary_index}}}$",
            xy=(boundary, 0),
            xytext=(10, -30),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=11,
            color="red",
        )

    _configure_ask_axis(ax, sorted_amplitudes)

    tick_positions = np.sort(
        np.concatenate((sorted_amplitudes, decision_boundaries))
    )
    ax.set_xticks(tick_positions)

    ax.set_title(
        f"{len(sorted_amplitudes)}-ary ASK Signal-Space Diagram "
        "With Decision Boundaries"
    )

    if output_path is not None:
        fig.savefig(
            output_path,
            format="svg",
            bbox_inches="tight",
        )

    plt.show()


plot_ask_signal_space(
    amplitudes=[0, 1 / 3, 2 / 3, 1],
    output_path="4_ask_signal_space.svg",
)

plot_ask_signal_space_decision_boundaries(
    amplitudes=[0, 1 / 3, 2 / 3, 1],
    output_path="4_ask_signal_space_decision_boundaries.svg",
)


import numpy as np


def plot_2fsk_signal_space(
    f0: float,
    f1: float,
    symbol_duration: float,
    sample_rate: float,
    amplitude: float = 1.0,
    output_path_signal_space: str | Path | None = None,
    output_path_decision_boundary: str | Path | None = None,
) -> None:
    """
    Plot the signal-space representation of coherent 2-FSK.

    The function supports both orthogonal and non-orthogonal tones. It derives
    an orthonormal 2D basis from the two actual time-domain waveforms using
    Gram-Schmidt orthogonalization.

    Parameters
    ----------
    f0, f1:
        Frequencies of the two FSK tones in Hz.

    symbol_duration:
        Duration of one FSK symbol in seconds.

    sample_rate:
        Sampling rate used to approximate the continuous-time inner products.

    amplitude:
        Tone amplitude.

    output_path_signal_space:
        Optional output path for the signal-space figure, e.g.
        "fsk_signal_space.svg".

    output_path_decision_boundary:
        Optional output path for the decision-boundary figure, e.g.
        "fsk_decision_boundary.svg".
    """
    if f0 <= 0 or f1 <= 0:
        raise ValueError("f0 and f1 must be positive.")
    if f0 == f1:
        raise ValueError("f0 and f1 must be different frequencies.")
    if symbol_duration <= 0:
        raise ValueError("symbol_duration must be positive.")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive.")
    if amplitude <= 0:
        raise ValueError("amplitude must be positive.")

    num_samples = int(round(symbol_duration * sample_rate))
    if num_samples < 2:
        raise ValueError(
            "symbol_duration * sample_rate must produce at least two samples."
        )

    t = np.arange(num_samples) / sample_rate
    dt = 1 / sample_rate

    # Two possible transmitted waveforms over one symbol interval.
    s0 = amplitude * np.cos(2 * np.pi * f0 * t)
    s1 = amplitude * np.cos(2 * np.pi * f1 * t)

    def inner_product(x: np.ndarray, y: np.ndarray) -> float:
        """Approximate the continuous-time inner product integral."""
        return float(np.sum(x * y) * dt)

    def norm(x: np.ndarray) -> float:
        return np.sqrt(inner_product(x, x))

    # Gram-Schmidt:
    # phi_1 points in the direction of s_0.
    phi_1 = s0 / norm(s0)

    # Remove from s_1 the component parallel to phi_1.
    s1_parallel_component = inner_product(s1, phi_1) * phi_1
    s1_perpendicular_component = s1 - s1_parallel_component

    perpendicular_norm = norm(s1_perpendicular_component)
    if perpendicular_norm < 1e-12:
        raise ValueError(
            "The two waveforms are numerically linearly dependent. "
            "Choose more distinct frequencies or a longer symbol duration."
        )

    phi_2 = s1_perpendicular_component / perpendicular_norm

    # Signal vectors in the orthonormal basis {phi_1, phi_2}.
    s0_vector = np.array([
        inner_product(s0, phi_1),
        inner_product(s0, phi_2),
    ])
    s1_vector = np.array([
        inner_product(s1, phi_1),
        inner_product(s1, phi_2),
    ])

    energy_0 = inner_product(s0, s0)
    energy_1 = inner_product(s1, s1)
    normalized_correlation = inner_product(s0, s1) / np.sqrt(energy_0 * energy_1)

    is_orthogonal = np.isclose(normalized_correlation, 0.0, atol=1e-3)
    orthogonality_label = "orthogonal" if is_orthogonal else "non-orthogonal"

    # Common plot limits.
    all_points = np.vstack([np.zeros(2), s0_vector, s1_vector])
    max_extent = np.max(np.abs(all_points))
    padding = max(0.25 * max_extent, 0.2)

    x_min = min(-padding, np.min(all_points[:, 0]) - padding)
    x_max = np.max(all_points[:, 0]) + padding
    y_min = min(-padding, np.min(all_points[:, 1]) - padding)
    y_max = np.max(all_points[:, 1]) + padding

    # -------------------------------------------------------------------------
    # Figure 1: signal-space vectors only.
    # -------------------------------------------------------------------------
    fig_signal, ax_signal = plt.subplots(figsize=(7, 6), constrained_layout=True)

    ax_signal.axhline(0, linewidth=0.8)
    ax_signal.axvline(0, linewidth=0.8)

    for vector, label in [
        (s0_vector, r"$\mathbf{s}_0$"),
        (s1_vector, r"$\mathbf{s}_1$"),
    ]:
        ax_signal.annotate(
            "",
            xy=vector,
            xytext=(0, 0),
            arrowprops={"arrowstyle": "->", "lw": 2},
        )
        ax_signal.scatter(*vector, s=90, zorder=3)
        ax_signal.annotate(
            label,
            xy=vector,
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=12,
        )

    ax_signal.set(
        xlim=(x_min, x_max),
        ylim=(y_min, y_max),
        xlabel=r"Coordinate along $\phi_1(t)$",
        ylabel=r"Coordinate along $\phi_2(t)$",
        title=(
            f"2-FSK signal space ({orthogonality_label})\n"
            rf"$\rho = {normalized_correlation:.3f}$"
        ),
    )
    ax_signal.set_aspect("equal", adjustable="box")
    ax_signal.grid(True, alpha=0.3)

    if output_path_signal_space is not None:
        fig_signal.savefig(output_path_signal_space, bbox_inches="tight")

    # -------------------------------------------------------------------------
    # Figure 2: ML decision boundary.
    #
    # For equally likely symbols in AWGN, the boundary is the perpendicular
    # bisector between s0 and s1:
    #
    #     ||r - s0||^2 = ||r - s1||^2
    # -------------------------------------------------------------------------
    fig_boundary, ax_boundary = plt.subplots(
        figsize=(7, 6),
        constrained_layout=True,
    )

    ax_boundary.axhline(0, linewidth=0.8)
    ax_boundary.axvline(0, linewidth=0.8)

    midpoint = (s0_vector + s1_vector) / 2
    connecting_vector = s1_vector - s0_vector

    # A vector perpendicular to the line joining the two signal points.
    boundary_direction = np.array([
        -connecting_vector[1],
        connecting_vector[0],
    ])
    boundary_direction /= np.linalg.norm(boundary_direction)

    line_length = 2.5 * max(x_max - x_min, y_max - y_min)
    boundary_points = np.vstack([
        midpoint - line_length * boundary_direction,
        midpoint + line_length * boundary_direction,
    ])

    ax_boundary.plot(
        boundary_points[:, 0],
        boundary_points[:, 1],
        linestyle="--",
        color="red",
        linewidth=2,
        label="ML decision boundary",
    )

    for vector, label in [
        (s0_vector, r"$\mathbf{s}_0$"),
        (s1_vector, r"$\mathbf{s}_1$"),
    ]:
        ax_boundary.scatter(*vector, s=90, zorder=3)
        ax_boundary.annotate(
            label,
            xy=vector,
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=12,
        )

    ax_boundary.text(
        x_min + 0.05 * (x_max - x_min),
        y_max - 0.10 * (y_max - y_min),
        r"Decide $s_0$",
        fontsize=12,
    )
    ax_boundary.text(
        x_max - 0.25 * (x_max - x_min),
        y_min + 0.08 * (y_max - y_min),
        r"Decide $s_1$",
        fontsize=12,
    )

    ax_boundary.set(
        xlim=(x_min, x_max),
        ylim=(y_min, y_max),
        xlabel=r"Coordinate along $\phi_1(t)$",
        ylabel=r"Coordinate along $\phi_2(t)$",
        title=(
            f"2-FSK ML decision boundary ({orthogonality_label})\n"
            rf"$\rho = {normalized_correlation:.3f}$"
        ),
    )
    ax_boundary.set_aspect("equal", adjustable="box")
    ax_boundary.grid(True, alpha=0.3)
    ax_boundary.legend()

    if output_path_decision_boundary is not None:
        fig_boundary.savefig(output_path_decision_boundary, bbox_inches="tight")

    plt.show()

# plot_2fsk_signal_space(
#     f0=1_000,
#     f1=1_100,
#     symbol_duration=0.01,
#     sample_rate=48_000,
#     output_path_signal_space="orthogonal_2fsk_signal_space.svg",
#     output_path_decision_boundary="orthogonal_2fsk_decision_boundary.svg",
# )
#
# plot_2fsk_signal_space(
#     f0=1_000,
#     f1=1_020,
#     symbol_duration=0.01,
#     sample_rate=48_000,
#     output_path_signal_space="nonorthogonal_2fsk_signal_space.svg",
#     output_path_decision_boundary="nonorthogonal_2fsk_decision_boundary.svg",
# )

import numpy as np
import matplotlib.pyplot as plt


import numpy as np
import matplotlib.pyplot as plt


def plot_psk_signal_space(
    order: int,
    amplitude: float = 1.0,
    output_path: str | None = None,
) -> None:
    """
    Plot M-PSK signal space and a separate decision-region plot.

    Parameters
    ----------
    order:
        Number of PSK symbols M, e.g. 2 for BPSK, 4 for QPSK, 8 for 8-PSK.

    amplitude:
        Radius of the constellation circle.

    output_path:
        Optional base SVG output path. For example, "qpsk.svg" creates
        "qpsk_constellation.svg" and "qpsk_decision_boundaries.svg".
    """
    if order < 2:
        raise ValueError("order must be at least 2.")

    symbol_indices = np.arange(order)
    phases = 2 * np.pi * symbol_indices / order

    i_values = amplitude * np.cos(phases)
    q_values = amplitude * np.sin(phases)

    label_distance = 0.21 * amplitude
    plot_limit = 1.45 * amplitude

    # ------------------------------------------------------------------
    # Plot 1: Signal-space constellation
    # ------------------------------------------------------------------
    fig_constellation, ax_constellation = plt.subplots(
        figsize=(6, 6),
        constrained_layout=True,
    )

    ax_constellation.scatter(i_values, q_values, s=100, zorder=3)

    circle = plt.Circle(
        (0, 0),
        amplitude,
        fill=False,
        linestyle="--",
        linewidth=1,
        alpha=0.6,
    )
    ax_constellation.add_patch(circle)

    for k, (i_value, q_value, phase) in enumerate(
        zip(i_values, q_values, phases)
    ):
        ax_constellation.annotate(
            rf"$m_{k}$",
            xy=(i_value, q_value),
            xytext=(
                i_value + label_distance * np.cos(phase),
                q_value + label_distance * np.sin(phase),
            ),
            ha="center",
            va="center",
        )

    _style_psk_axes(
        ax_constellation,
        order=order,
        plot_limit=plot_limit,
        title=rf"${order}$-PSK Signal Space",
    )

    # ------------------------------------------------------------------
    # Plot 2: Signal space with maximum-likelihood decision boundaries
    # ------------------------------------------------------------------
    fig_boundaries, ax_boundaries = plt.subplots(
        figsize=(6, 6),
        constrained_layout=True,
    )

    ax_boundaries.scatter(i_values, q_values, s=100, zorder=3)

    circle = plt.Circle(
        (0, 0),
        amplitude,
        fill=False,
        linestyle="--",
        linewidth=1,
        alpha=0.6,
    )
    ax_boundaries.add_patch(circle)

    for k, (i_value, q_value, phase) in enumerate(
        zip(i_values, q_values, phases)
    ):
        ax_boundaries.annotate(
            rf"$m_{k}$",
            xy=(i_value, q_value),
            xytext=(
                i_value + label_distance * np.cos(phase),
                q_value + label_distance * np.sin(phase),
            ),
            ha="center",
            va="center",
        )

    # Boundaries lie halfway in angle between adjacent PSK symbols.
    boundary_phases = phases + np.pi / order

    for boundary_phase in boundary_phases:
        x_end = plot_limit * np.cos(boundary_phase)
        y_end = plot_limit * np.sin(boundary_phase)

        ax_boundaries.plot(
            [0, x_end],
            [0, y_end],
            color="red",
            linewidth=1.5,
            zorder=2,
        )

    _style_psk_axes(
        ax_boundaries,
        order=order,
        plot_limit=plot_limit,
        title=rf"${order}$-PSK Signal Space with Decision Boundaries",
    )

    if output_path is not None:
        base_path = output_path.removesuffix(".svg")

        fig_constellation.savefig(
            f"{base_path}_constellation.svg",
            bbox_inches="tight",
        )
        fig_boundaries.savefig(
            f"{base_path}_decision_boundaries.svg",
            bbox_inches="tight",
        )

    plt.show()


def _style_psk_axes(
    ax: plt.Axes,
    order: int,
    plot_limit: float,
    title: str,
) -> None:
    ax.axhline(0, linewidth=1)
    ax.axvline(0, linewidth=1)

    ax.set_xlim(-plot_limit, plot_limit)
    ax.set_ylim(-plot_limit, plot_limit)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"In-phase component $I$")
    ax.set_ylabel(r"Quadrature component $Q$")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

# plot_psk_signal_space(
#     order=4,
#     output_path="qpsk_signal_space.svg",
# )


import numpy as np
import matplotlib.pyplot as plt


def plot_16qam_grid(
    spacing: float = 2.0,
    output_path_constellation: str | None = None,
    output_path_boundaries: str | None = None,
) -> None:
    """
    Plot a standard square 16-QAM constellation in two separate figures:
    1. Constellation points only
    2. Constellation points with ML decision boundaries

    Parameters
    ----------
    spacing:
        Distance between adjacent I/Q amplitude levels.

    output_path_constellation:
        Optional output path for the constellation-only figure.

    output_path_boundaries:
        Optional output path for the constellation-with-boundaries figure.
    """
    levels = np.array([-3, -1, 1, 3], dtype=float) * (spacing / 2)

    i_values, q_values = np.meshgrid(levels, levels)
    i_values = i_values.ravel()
    q_values = q_values.ravel()

    padding = spacing
    x_limits = (levels.min() - padding, levels.max() + padding)
    y_limits = (levels.min() - padding, levels.max() + padding)

    # ------------------------------------------------------------------
    # Plot 1: constellation only
    # ------------------------------------------------------------------
    fig1, ax1 = plt.subplots(figsize=(6, 6), constrained_layout=True)

    ax1.scatter(i_values, q_values, s=100, zorder=3)

    for index, (i, q) in enumerate(zip(i_values, q_values)):
        ax1.annotate(
            rf"$m_{{{index}}}$",
            xy=(i, q),
            xytext=(6, 6),
            textcoords="offset points",
        )

    ax1.axhline(0, color="black", linewidth=1)
    ax1.axvline(0, color="black", linewidth=1)

    ax1.set_xticks(levels)
    ax1.set_yticks(levels)

    ax1.set_xlim(*x_limits)
    ax1.set_ylim(*y_limits)

    ax1.set_aspect("equal", adjustable="box")
    ax1.grid(True, linestyle="--", alpha=0.4)

    ax1.set_xlabel(r"In-phase amplitude $A_I$")
    ax1.set_ylabel(r"Quadrature amplitude $A_Q$")
    ax1.set_title("16-QAM Signal Space")

    if output_path_constellation is not None:
        fig1.savefig(output_path_constellation, bbox_inches="tight")

    # ------------------------------------------------------------------
    # Plot 2: constellation with decision boundaries
    # ------------------------------------------------------------------
    fig2, ax2 = plt.subplots(figsize=(6, 6), constrained_layout=True)

    ax2.scatter(i_values, q_values, s=100, zorder=3)

    for index, (i, q) in enumerate(zip(i_values, q_values)):
        ax2.annotate(
            rf"$m_{{{index}}}$",
            xy=(i, q),
            xytext=(6, 6),
            textcoords="offset points",
        )

    boundaries = (levels[:-1] + levels[1:]) / 2

    for boundary in boundaries:
        ax2.axvline(boundary, color="red", linewidth=1.5, zorder=2)
        ax2.axhline(boundary, color="red", linewidth=1.5, zorder=2)

    ax2.axhline(0, color="black", linewidth=1)
    ax2.axvline(0, color="black", linewidth=1)

    ax2.set_xticks(levels)
    ax2.set_yticks(levels)

    ax2.set_xlim(*x_limits)
    ax2.set_ylim(*y_limits)

    ax2.set_aspect("equal", adjustable="box")
    ax2.grid(True, linestyle="--", alpha=0.4)

    ax2.set_xlabel(r"In-phase amplitude $A_I$")
    ax2.set_ylabel(r"Quadrature amplitude $A_Q$")
    ax2.set_title("16-QAM Signal Space with Decision Boundaries")

    if output_path_boundaries is not None:
        fig2.savefig(output_path_boundaries, bbox_inches="tight")

    plt.show()


# plot_16qam_grid(
#     output_path_constellation="16qam_signal_space.svg",
#     output_path_boundaries="16qam_decision_boundaries.svg",
# )


import numpy as np
import matplotlib.pyplot as plt


def plot_analog_and_digitized_sine(
    frequency: float = 2.0,
    amplitude: float = 1.0,
    duration: float = 1.0,
    analog_sample_rate: int = 10_000,
    sampling_rate: int = 16,
    quantization_bits: int = 3,
    output_path: str | None = None,
) -> None:
    """
    Plot an analog sine wave alongside its sampled and quantized representation.

    The plot includes:
    - the ideal continuous-time sine wave,
    - sampled values before quantization,
    - quantized digital samples,
    - vertical sampling error,
    - vertical quantization error.

    Parameters
    ----------
    frequency:
        Sine-wave frequency in Hz.

    amplitude:
        Peak amplitude of the analog sine wave.

    duration:
        Signal duration in seconds.

    analog_sample_rate:
        Dense plotting resolution used to approximate the continuous-time signal.

    sampling_rate:
        ADC sampling rate in Hz.

    quantization_bits:
        Number of uniform quantization bits.

    output_path:
        Optional SVG output path, for example:
        "analog_vs_digital.svg".
    """
    if frequency <= 0:
        raise ValueError("frequency must be positive.")

    if amplitude <= 0:
        raise ValueError("amplitude must be positive.")

    if duration <= 0:
        raise ValueError("duration must be positive.")

    if sampling_rate <= 0:
        raise ValueError("sampling_rate must be positive.")

    if quantization_bits < 1:
        raise ValueError("quantization_bits must be at least 1.")

    if analog_sample_rate < sampling_rate:
        raise ValueError(
            "analog_sample_rate should be at least as large as sampling_rate."
        )

    # Dense time grid: visual approximation of the analog signal.
    t_analog = np.linspace(
        0,
        duration,
        int(duration * analog_sample_rate),
        endpoint=False,
    )
    analog_signal = amplitude * np.sin(2 * np.pi * frequency * t_analog)

    # Sampled signal before quantization.
    t_samples = np.arange(0, duration, 1 / sampling_rate)
    sampled_signal = amplitude * np.sin(2 * np.pi * frequency * t_samples)

    # Uniform mid-tread quantizer with symmetric levels around zero.
    num_levels = 2**quantization_bits
    quantization_step = 2 * amplitude / (num_levels - 1)

    quantized_signal = (
        np.round(sampled_signal / quantization_step) * quantization_step
    )
    quantized_signal = np.clip(
        quantized_signal,
        -amplitude,
        amplitude,
    )

    fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)

    # Analog waveform.
    ax.plot(
        t_analog,
        analog_signal,
        label="Analogue sine wave",
        linewidth=2,
        zorder=1,
    )

    # Sampled values before quantization.
    ax.scatter(
        t_samples,
        sampled_signal,
        label="Sampled values",
        s=45,
        zorder=4,
    )

    # Quantized digital values as a staircase.
    ax.step(
        t_samples,
        quantized_signal,
        where="mid",
        label="Quantized digital signal",
        linewidth=2,
        zorder=2,
    )

    ax.scatter(
        t_samples,
        quantized_signal,
        s=45,
        zorder=5,
    )

    # Quantization error at each sampling instant.
    ax.vlines(
        t_samples,
        sampled_signal,
        quantized_signal,
        linewidth=1.5,
        label="Quantization error",
        zorder=3,
    )

    # Show quantization levels.
    quantization_levels = np.linspace(-amplitude, amplitude, num_levels)
    for level in quantization_levels:
        ax.axhline(
            level,
            linewidth=0.8,
            linestyle=":",
            alpha=0.35,
            zorder=0,
        )

    ax.set_title(
        "Analogue Sine Wave, Sampling, and Quantization"
    )
    ax.set_xlabel("Time $t$ [s]")
    ax.set_ylabel("Amplitude")

    ax.set_xlim(0, duration)
    ax.set_ylim(-1.2 * amplitude, 1.2 * amplitude)

    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")

    if output_path is not None:
        fig.savefig(
            output_path,
            format="svg",
            bbox_inches="tight",
        )

    plt.show()

# plot_analog_and_digitized_sine(output_path="digital-analog-comparison.svg")

import numpy as np
import matplotlib.pyplot as plt


import numpy as np
import matplotlib.pyplot as plt


def signed_alias_frequency(f: float, fs: float) -> float:
    """
    Fold frequency f into the Nyquist interval [-fs/2, fs/2).
    """
    return ((f + fs / 2) % fs) - fs / 2


def plot_aliasing(
    signal_frequency: float = 7.0,
    sampling_frequency: float = 10.0,
    duration: float = 1.0,
    output_path: str = "aliasing_sine_wave.svg",
) -> None:
    nyquist_frequency = sampling_frequency / 2

    f_alias_signed = signed_alias_frequency(
        signal_frequency,
        sampling_frequency,
    )

    f_alias_magnitude = abs(f_alias_signed)

    # Dense time axis for visually continuous curves
    t = np.linspace(0, duration, 5000)

    # Sample times
    n = np.arange(0, int(duration * sampling_frequency) + 1)
    t_samples = n / sampling_frequency

    # Original continuous-time signal
    x = np.sin(2 * np.pi * signal_frequency * t)

    # Correct alias signal, including sign/phase
    x_alias = np.sin(2 * np.pi * f_alias_signed * t)

    # Samples taken from the original signal
    x_samples = np.sin(2 * np.pi * signal_frequency * t_samples)

    fig, ax = plt.subplots(figsize=(10, 4.8))

    ax.plot(
        t,
        x,
        linewidth=1.5,
        label=rf"Original sine: $f = {signal_frequency:g}\,\mathrm{{Hz}}$",
    )

    ax.plot(
        t,
        x_alias,
        linestyle="--",
        linewidth=2,
        label=rf"Alias sine: $f = {f_alias_signed:g}\,\mathrm{{Hz}}$ "
              rf"$(|f| = {f_alias_magnitude:g}\,\mathrm{{Hz}})$",
    )

    ax.stem(
        t_samples,
        x_samples,
        linefmt=":",
        markerfmt="o",
        basefmt=" ",
        label=rf"Samples: $f_s = {sampling_frequency:g}\,\mathrm{{Hz}}$",
    )

    ax.set_title(
        rf"Aliasing: $f = {signal_frequency:g}\,\mathrm{{Hz}}$ sampled at "
        rf"$f_s = {sampling_frequency:g}\,\mathrm{{Hz}}$"
    )

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Amplitude")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")



    ax.set_ylim(-1.5, 1.5)

    fig.tight_layout()
    fig.savefig(output_path, format="svg", bbox_inches="tight")
    plt.show()

#plot_aliasing()

import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import spectrogram


def plot_shifted_chirp_spectrogram(
    sample_rate=16_000,
    symbol_duration=0.5,
    start_frequency=2_000,
    end_frequency=6_000,
    amplitude=0.8,
    shift_fraction=0.5,
    output_path="chirp_time_shift_spectrogram.svg",
):
    samples_per_symbol = int(round(sample_rate * symbol_duration))
    t = np.arange(samples_per_symbol) / sample_rate

    bandwidth = end_frequency - start_frequency
    chirp_rate = bandwidth / symbol_duration

    def create_chirp(time_shift):
        shifted_time = (t + time_shift) % symbol_duration

        phase = 2 * np.pi * (
            start_frequency * shifted_time
            + 0.5 * chirp_rate * shifted_time**2
        )

        return amplitude * np.cos(phase)

    normal_chirp = create_chirp(0.0)
    shifted_chirp = create_chirp(shift_fraction * symbol_duration)

    signal = np.concatenate([
        normal_chirp,
        shifted_chirp,
    ])

    frequencies, times, spectrum = spectrogram(
        signal,
        fs=sample_rate,
        window="hann",
        nperseg=256,
        noverlap=192,
        nfft=512,
        mode="magnitude",
    )

    spectrum_db = 20 * np.log10(spectrum + 1e-12)

    fig, ax = plt.subplots(
        figsize=(10, 4.5),
        constrained_layout=True,
    )

    mesh = ax.pcolormesh(
        times,
        frequencies,
        spectrum_db,
        shading="auto",
        rasterized=True,
    )

    ax.axvline(
        symbol_duration,
        color="red",
        linewidth=2,
        label="Symbol boundary",
    )

    ax.set_xlim(0, 2 * symbol_duration)
    ax.set_ylim(0, end_frequency + 500)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Frequency [Hz]")
    ax.set_title("Normal and cyclically shifted up-chirp")
    ax.set_xticks([
        0,
        symbol_duration,
        2 * symbol_duration,
    ])
    ax.set_xticklabels([
        "0",
        "$T_s$",
        "$2T_s$",
    ])
    ax.legend()

    colorbar = fig.colorbar(mesh, ax=ax)
    colorbar.set_label("Magnitude [dB]")

    fig.savefig(
        output_path,
        format="svg",
        bbox_inches="tight",
    )

    plt.show()


#plot_shifted_chirp_spectrogram()

import numpy as np
import matplotlib.pyplot as plt

from acoustic_toolbox.standards import iso_9613_1_1993 as iso9613


def plot_geometric_spreading(
    reference_distance=1.0,
    reference_level=0.0,
    min_distance=1.0,
    max_distance=100.0,
    num_distances=2000,
    output_path="geometric_spreading.svg",
):
    if reference_distance <= 0:
        raise ValueError("reference_distance must be greater than zero")

    if min_distance <= 0:
        raise ValueError("min_distance must be greater than zero")

    distances = np.linspace(
        min_distance,
        max_distance,
        num_distances,
    )

    sound_pressure_levels = (
        reference_level
        - 20.0 * np.log10(distances / reference_distance)
    )

    fig, ax = plt.subplots(
        figsize=(10, 5),
        constrained_layout=True,
    )

    ax.plot(
        distances,
        sound_pressure_levels,
    )

    ax.axvline(
        reference_distance,
        linestyle="--",
        label=f"Reference distance: {reference_distance:g} m",
    )

    ax.set_xlabel("Distance [m]")
    ax.set_ylabel("Sound pressure level relative to reference [dB]")
    ax.set_title("Geometric spreading of sound")
    ax.grid(True)
    ax.legend()
    ax.set_xlim(min_distance, max_distance)

    fig.savefig(
        output_path,
        format="svg",
        bbox_inches="tight",
    )

    plt.show()


def plot_sound_attenuation(
    temperature_celsius=20.0,
    relative_humidity=0.5,
    pressure_kpa=101.325,
    distances=(1, 10, 100),
    min_frequency=100,
    max_frequency=22000,
    num_frequencies=2000,
    output_path="atmospheric_sound_attenuation.svg",
):
    temperature_kelvin = temperature_celsius + 273.15

    frequencies = np.linspace(
        min_frequency,
        max_frequency,
        num_frequencies,
    )

    saturation_pressure = iso9613.saturation_pressure(
        temperature_kelvin,
    )

    water_vapour_concentration = (
        iso9613.molar_concentration_water_vapour(
            relative_humidity=relative_humidity,
            saturation_pressure=saturation_pressure,
            pressure=pressure_kpa,
        )
    )

    oxygen_relaxation_frequency = (
        iso9613.relaxation_frequency_oxygen(
            pressure=pressure_kpa,
            h=water_vapour_concentration,
        )
    )

    nitrogen_relaxation_frequency = (
        iso9613.relaxation_frequency_nitrogen(
            pressure=pressure_kpa,
            temperature=temperature_kelvin,
            h=water_vapour_concentration,
        )
    )

    attenuation_per_meter = iso9613.attenuation_coefficient(
        pressure=pressure_kpa,
        temperature=temperature_kelvin,
        reference_pressure=iso9613.REFERENCE_PRESSURE,
        reference_temperature=iso9613.REFERENCE_TEMPERATURE,
        relaxation_frequency_nitrogen=nitrogen_relaxation_frequency,
        relaxation_frequency_oxygen=oxygen_relaxation_frequency,
        frequency=frequencies,
    )

    fig, ax = plt.subplots(
        figsize=(10, 5),
        constrained_layout=True,
    )

    for distance in distances:
        attenuation = attenuation_per_meter * distance

        ax.plot(
            frequencies,
            attenuation,
            label=f"{distance:g} m",
        )

    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Atmospheric absorption loss [dB]")
    ax.set_title(
        f"Atmospheric sound absorption at "
        f"{temperature_celsius:g} °C and "
        f"{relative_humidity * 100:g}% relative humidity"
    )
    ax.grid(True)
    ax.legend(title="Distance")
    ax.set_xlim(min_frequency, max_frequency)
    ax.set_ylim(bottom=0)

    fig.savefig(
        output_path,
        format="svg",
        bbox_inches="tight",
    )

    plt.show()


def plot_combined_sound_attenuation(
    temperature_celsius=20.0,
    relative_humidity=0.5,
    pressure_kpa=101.325,
    reference_distance=1.0,
    distances=(10, 100),
    min_frequency=100,
    max_frequency=22000,
    num_frequencies=2000,
    output_path="combined_sound_attenuation.svg",
):
    if reference_distance <= 0:
        raise ValueError("reference_distance must be greater than zero")

    if any(distance <= 0 for distance in distances):
        raise ValueError("All distances must be greater than zero")

    temperature_kelvin = temperature_celsius + 273.15

    frequencies = np.linspace(
        min_frequency,
        max_frequency,
        num_frequencies,
    )

    saturation_pressure = iso9613.saturation_pressure(
        temperature_kelvin,
    )

    water_vapour_concentration = (
        iso9613.molar_concentration_water_vapour(
            relative_humidity=relative_humidity,
            saturation_pressure=saturation_pressure,
            pressure=pressure_kpa,
        )
    )

    oxygen_relaxation_frequency = (
        iso9613.relaxation_frequency_oxygen(
            pressure=pressure_kpa,
            h=water_vapour_concentration,
        )
    )

    nitrogen_relaxation_frequency = (
        iso9613.relaxation_frequency_nitrogen(
            pressure=pressure_kpa,
            temperature=temperature_kelvin,
            h=water_vapour_concentration,
        )
    )

    attenuation_per_meter = iso9613.attenuation_coefficient(
        pressure=pressure_kpa,
        temperature=temperature_kelvin,
        reference_pressure=iso9613.REFERENCE_PRESSURE,
        reference_temperature=iso9613.REFERENCE_TEMPERATURE,
        relaxation_frequency_nitrogen=nitrogen_relaxation_frequency,
        relaxation_frequency_oxygen=oxygen_relaxation_frequency,
        frequency=frequencies,
    )

    fig, ax = plt.subplots(
        figsize=(10, 5),
        constrained_layout=True,
    )

    for distance in distances:
        geometric_loss = (
            20.0
            * np.log10(distance / reference_distance)
        )

        atmospheric_loss = (
            attenuation_per_meter
            * (distance - reference_distance)
        )

        total_loss = geometric_loss + atmospheric_loss

        ax.plot(
            frequencies,
            total_loss,
            label=f"Total loss at {distance:g} m",
        )

        ax.axhline(
            geometric_loss,
            linestyle="--",
            label=f"Geometric loss at {distance:g} m",
        )

    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Attenuation relative to reference distance [dB]")
    ax.set_title(
        f"Geometric spreading and atmospheric absorption "
        f"relative to {reference_distance:g} m"
    )
    ax.grid(True)
    ax.legend()
    ax.set_xlim(min_frequency, max_frequency)
    ax.set_ylim(bottom=0)

    fig.savefig(
        output_path,
        format="svg",
        bbox_inches="tight",
    )

    plt.show()

#
# plot_geometric_spreading()
#
# plot_sound_attenuation()
#
# plot_combined_sound_attenuation(
#     reference_distance=1,
#     distances=(10, 100),
# )

def plot_sinusoid_to_iq_representation(
    amplitude=1.0,
    phase=np.pi / 4,
    carrier_frequency=1.0,
    output_path="sinusoid_to_iq_representation.svg",
):
    I = amplitude * np.cos(phase)
    Q = amplitude * np.sin(phase)

    t = np.linspace(0, 2 / carrier_frequency, 1000)
    s = amplitude * np.cos(2 * np.pi * carrier_frequency * t + phase)
    i_component = I * np.cos(2 * np.pi * carrier_frequency * t)
    q_component = -Q * np.sin(2 * np.pi * carrier_frequency * t)

    fig = plt.figure(figsize=(9, 6), constrained_layout=True)
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.0])

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(t, s, linewidth=2)
    ax1.axhline(0, linewidth=1)
    ax1.set_xlim(t[0], t[-1])
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Amplitude")
    ax1.set_title("Sinusoidal signal")
    ax1.text(
        0.02,
        0.92,
        rf"$s(t)=A\cos(2\pi f_c t+\phi)$",
        transform=ax1.transAxes,
        ha="left",
        va="top",
    )

    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(t, i_component, linewidth=2, label=rf"$I\cos(2\pi f_c t)$")
    ax2.plot(t, q_component, linewidth=2, label=rf"$-Q\sin(2\pi f_c t)$")
    ax2.plot(t, s, linewidth=2, linestyle="--", label=rf"$s(t)$")
    ax2.axhline(0, linewidth=1)
    ax2.set_xlim(t[0], t[-1])
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Amplitude")
    ax2.set_title("Decomposition into orthogonal components")
    ax2.legend()

    fig.savefig(output_path, format="svg", bbox_inches="tight")
    plt.show()


plot_sinusoid_to_iq_representation()

