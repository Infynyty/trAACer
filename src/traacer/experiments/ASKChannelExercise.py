from __future__ import annotations

from typing import Literal

import numpy as np


SAMPLE_RATE = 48_000
CARRIER_FREQUENCY_HZ = 1_000.0
SYMBOL_DURATION_SECONDS = 0.01
ASK_AMPLITUDES = np.array([0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0])

Demodulator = Literal["Direct comparison", "Coherent", "Incoherent"]


def parse_bits(text: str) -> np.ndarray:
    """Parse a comma- or whitespace-separated sequence of 4-ASK bits."""

    tokens = text.replace(",", " ").split()
    if not tokens:
        raise ValueError("Enter at least one pair of bits")

    try:
        bits = np.array([int(token) for token in tokens], dtype=np.uint8)
    except ValueError as error:
        raise ValueError("Bits must be separated by spaces or commas") from error

    if np.any((bits != 0) & (bits != 1)):
        raise ValueError("Bits must only contain 0 and 1")
    if len(bits) % 2:
        raise ValueError("4-ASK requires an even number of bits")

    return bits


def _bits_to_symbol_indices(bits: np.ndarray) -> np.ndarray:
    bits = np.asarray(bits, dtype=np.uint8).reshape(-1)
    if not len(bits) or len(bits) % 2:
        raise ValueError("4-ASK requires a non-empty, even number of bits")
    if np.any((bits != 0) & (bits != 1)):
        raise ValueError("Bits must only contain 0 and 1")

    return bits.reshape(-1, 2) @ np.array([2, 1], dtype=np.uint8)


def _carrier(phase_radians: float = 0.0) -> np.ndarray:
    samples_per_symbol = round(SAMPLE_RATE * SYMBOL_DURATION_SECONDS)
    time = np.arange(samples_per_symbol, dtype=np.float64) / SAMPLE_RATE
    return np.sin(
        2.0 * np.pi * CARRIER_FREQUENCY_HZ * time + phase_radians
    )


def modulate_4ask(
    bits: np.ndarray,
    *,
    phase_radians: float = 0.0,
) -> np.ndarray:
    """Turn bit pairs into 4-ASK symbols using the requested carrier phase."""

    amplitudes = ASK_AMPLITUDES[_bits_to_symbol_indices(bits)]
    return (amplitudes[:, np.newaxis] * _carrier(phase_radians)).reshape(-1)


def simulate_channel(
    bits: np.ndarray,
    *,
    awgn_standard_deviation: float = 0.0,
    attenuation: float = 1.0,
    phase_shift_degrees: float = 0.0,
    random_seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the transmitted and channel-impaired 4-ASK waveforms."""

    if awgn_standard_deviation < 0.0:
        raise ValueError("AWGN standard deviation cannot be negative")
    if not 0.0 <= attenuation <= 1.0:
        raise ValueError("Attenuation must be between 0 and 1")

    transmitted = modulate_4ask(bits)
    phase_radians = np.deg2rad(phase_shift_degrees)
    received = attenuation * modulate_4ask(
        bits,
        phase_radians=phase_radians,
    )

    if awgn_standard_deviation:
        generator = np.random.default_rng(random_seed)
        received = received + generator.normal(
            loc=0.0,
            scale=awgn_standard_deviation,
            size=len(received),
        )

    return transmitted, received


def _candidate_symbols() -> np.ndarray:
    return ASK_AMPLITUDES[:, np.newaxis] * _carrier()


def demodulate_4ask(
    received: np.ndarray,
    method: Demodulator,
) -> np.ndarray:
    """Decode 4-ASK symbols, using -1 when direct comparison finds no match."""

    samples = np.asarray(received, dtype=np.float64).reshape(-1)
    samples_per_symbol = len(_carrier())
    if not len(samples) or len(samples) % samples_per_symbol:
        raise ValueError("Received signal must contain complete ASK symbols")

    symbols = samples.reshape(-1, samples_per_symbol)
    candidates = _candidate_symbols()

    if method == "Direct comparison":
        return np.array(
            [
                next(
                    (
                        index
                        for index, candidate in enumerate(candidates)
                        if np.array_equal(symbol, candidate)
                    ),
                    -1,
                )
                for symbol in symbols
            ],
            dtype=np.int8,
        )

    if method == "Coherent":
        squared_distances = np.sum(
            (symbols[:, np.newaxis, :] - candidates[np.newaxis, :, :]) ** 2,
            axis=2,
        )
        return np.argmin(squared_distances, axis=1).astype(np.int8)

    if method == "Incoherent":
        in_phase_template = _carrier()
        quadrature_template = _carrier(np.pi / 2.0)
        in_phase = (
            symbols @ in_phase_template
            / np.dot(in_phase_template, in_phase_template)
        )
        quadrature = (
            symbols @ quadrature_template
            / np.dot(quadrature_template, quadrature_template)
        )
        estimated_amplitudes = np.hypot(in_phase, quadrature)
        return np.argmin(
            np.abs(
                estimated_amplitudes[:, np.newaxis]
                - ASK_AMPLITUDES[np.newaxis, :]
            ),
            axis=1,
        ).astype(np.int8)

    raise ValueError(f"Unknown demodulator: {method}")


def symbol_indices_to_text(symbol_indices: np.ndarray) -> str:
    """Format decoded symbols as bits while preserving failed exact matches."""

    pairs = ("00", "01", "10", "11")
    return " ".join(
        pairs[int(index)] if index >= 0 else "??"
        for index in symbol_indices
    )


def _count_bit_errors(
    transmitted_bits: np.ndarray,
    decoded_symbols: np.ndarray,
) -> int:
    transmitted_symbols = _bits_to_symbol_indices(transmitted_bits)
    errors = 0
    for transmitted, decoded in zip(transmitted_symbols, decoded_symbols):
        if decoded < 0:
            errors += 2
        else:
            errors += (int(transmitted) ^ int(decoded)).bit_count()
    return errors


async def main() -> None:
    """Display the interactive channel and demodulation exercise."""

    import matplotlib.pyplot as plt
    import ipywidgets as widgets
    from IPython.display import display

    bits_input = widgets.Text(
        value="0 0 0 1 1 0 1 1",
        description="Bits",
        layout=widgets.Layout(width="420px"),
    )
    demodulator = widgets.Dropdown(
        options=("Direct comparison", "Coherent", "Incoherent"),
        value="Direct comparison",
        description="Demodulator",
        style={"description_width": "initial"},
    )
    awgn = widgets.FloatSlider(
        value=0.0,
        min=0.0,
        max=0.5,
        step=0.01,
        description="AWGN (standard deviation)",
        readout_format=".2f",
        continuous_update=False,
        style={"description_width": "initial"},
    )
    attenuation = widgets.FloatSlider(
        value=1.0,
        min=0.0,
        max=1.0,
        step=0.05,
        description="Attenuation (channel gain)",
        readout_format=".2f",
        continuous_update=False,
        style={"description_width": "initial"},
    )
    phase_shift = widgets.IntSlider(
        value=0,
        min=-180,
        max=180,
        step=5,
        description="Phase shift [degrees]",
        continuous_update=False,
        style={"description_width": "initial"},
    )
    random_seed = widgets.IntText(
        value=0,
        description="Noise seed",
    )
    run_button = widgets.Button(
        description="Transmit and decode",
        button_style="primary",
    )
    output = widgets.Output()

    def run_experiment(_: widgets.Button | None = None) -> None:
        with output:
            output.clear_output(wait=True)
            try:
                bits = parse_bits(bits_input.value)
                transmitted, received = simulate_channel(
                    bits,
                    awgn_standard_deviation=awgn.value,
                    attenuation=attenuation.value,
                    phase_shift_degrees=phase_shift.value,
                    random_seed=random_seed.value,
                )
                decoded = demodulate_4ask(received, demodulator.value)
            except ValueError as error:
                print(error)
                return

            transmitted_text = " ".join(
                f"{first}{second}"
                for first, second in bits.reshape(-1, 2)
            )
            decoded_text = symbol_indices_to_text(decoded)
            bit_errors = _count_bit_errors(bits, decoded)

            print(f"Transmitted: {transmitted_text}")
            print(f"Decoded:     {decoded_text}")
            print(f"Bit errors:  {bit_errors} / {len(bits)}")
            print(
                "Result:      "
                + ("correct" if bit_errors == 0 else "decoding failed")
            )

            time_ms = np.arange(len(transmitted)) / SAMPLE_RATE * 1_000.0
            fig, axis = plt.subplots(
                figsize=(12, 4),
                constrained_layout=True,
            )
            axis.plot(time_ms, transmitted, label="Transmitted", alpha=0.8)
            axis.plot(time_ms, received, label="Received", alpha=0.75)
            for boundary in range(len(decoded) + 1):
                axis.axvline(
                    boundary * SYMBOL_DURATION_SECONDS * 1_000.0,
                    color="black",
                    linestyle="--",
                    alpha=0.2,
                )
            axis.set_xlabel("Time [ms]")
            axis.set_ylabel("Amplitude")
            axis.set_title(f"4-ASK channel — {demodulator.value}")
            axis.grid(True, alpha=0.25)
            axis.legend()
            plt.show()

    run_button.on_click(run_experiment)
    display(
        widgets.VBox(
            [
                bits_input,
                demodulator,
                awgn,
                attenuation,
                phase_shift,
                random_seed,
                run_button,
                output,
            ]
        )
    )
    run_experiment()


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
