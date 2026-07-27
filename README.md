# trAACer

trAACer is a Python toolkit for experimenting with digital communication over
an acoustic channel. It turns bits or application data into audio, plays the
signal through a computer speaker, records it with a browser-enabled device,
and streams the recording back for synchronization, demodulation, decoding,
and analysis.

The project includes:

- ASK, OOK, FSK, PSK, QAM, and OFDM modulation components
- framing, preamble detection, and error-correction stages
- browser-based microphone capture over HTTPS and WebSockets
- WAV-file and local sound-device adapters
- BER, frequency-response, channel-capacity, and AEC experiments
- CSV and SVG measurement output

## How it works

Most experiments use a computer and a second device, such as a phone:

```text
payload -> framing -> error correction -> modulation -> computer speaker
                                                               |
                                                               v
results <- metrics <- decoding <- demodulation <- phone microphone
                                                   |
                                             HTTPS/WebSocket
```

The Python process hosts a local HTTPS page. The page captures unprocessed
microphone samples in the browser and streams them back to Python as PCM audio.
Using a separate device avoids coupling playback and recording to the same
computer, although the AEC experiment can also test a local duplex setup.

## Requirements

- Python 3.10 or newer
- A working speaker or audio output device
- A phone, tablet, or second computer with a microphone and modern browser
- Both devices on the same local network
- Permission to listen on TCP port `8000`

Some experiments open Matplotlib windows, so a graphical desktop is useful.
The AEC experiment additionally needs a local input device.

## Setup

From the repository root, create an isolated environment and install the
package in editable mode:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install requests
```

On Windows PowerShell, activate the environment with:

```powershell
.\.venv\Scripts\Activate.ps1
```

On systems where `python` already refers to Python 3.10 or newer, the included
Makefile provides a shortcut:

```bash
make install
source .venv/bin/activate
python -m pip install requests
```

Check the installation without accessing any audio hardware:

```bash
python -c "import traacer; print('trAACer is installed')"
python -m compileall -q src
```

Keep the virtual environment active for all commands below.

## Quick start

The continuous ASK example is a simple way to verify the complete
speaker-to-browser-microphone path:

```bash
python -m traacer.experiments.ASKInfinite
```

The command starts the HTTPS server and prints a local URL and QR code.

1. Put the computer and recording device on the same network.
2. Scan the QR code or open the printed `https://<computer-ip>:8000` URL on
   the recording device. Do not use `127.0.0.1` on the second device.
3. Accept the development-certificate warning if the browser displays one.
4. Select **Manual start** and allow microphone access.
5. Keep the microphone close to the speaker. A rolling plot on the computer
   shows the received ASK signal.
6. Stop the experiment with `Ctrl+C`.

The local certificate and key live in
`src/traacer/receiver/certs/`. They are intended for development on a trusted
local network, not for a public deployment.

## Running experiments

Experiments are Python modules under `src/traacer/experiments`. Run them from
the repository root with `python -m`, preserving the filename's capitalization.
Only run one browser-backed experiment at a time because they normally share
port `8000`.

Useful starting points include:

| Module | Purpose |
| --- | --- |
| `ASKInfinite` | Continuously transmit a fixed ASK sequence and plot received audio |
| `OOKTest` | Measure OOK BER while reducing symbol duration |
| `ASKOrderTest` | Compare BER across ASK modulation orders |
| `OFDMSubcarrierTest` | Compare OFDM BER across subcarrier counts |
| `FrequencyResponse` | Measure the acoustic channel's frequency response |
| `ChannelCapacityEstimation` | Estimate channel capacity from a measured signal and noise |
| `ChannelCodingExercise` | Compare channel-coding configurations |
| `ModulationOrderExercise` | Explore the modulation-order trade-off |
| `OFDMQAMExercise` | Explore QAM order and OFDM parameters |
| `AECExperiment` | Compare local duplex and external-microphone recordings |

Ask an experiment for its supported options:

```bash
python -m traacer.experiments.OOKTest --help
python -m traacer.experiments.ASKOrderTest --help
python -m traacer.experiments.OFDMSubcarrierTest --help
python -m traacer.experiments.AECExperiment --help
```

For example, run an OOK sweep at 8 kHz, write a CSV file, and skip the
interactive plot:

```bash
python -m traacer.experiments.OOKTest \
  --frequency-hz 8000 \
  --csv measurements/ook-ber.csv \
  --no-plot
```

List the sound devices available to the AEC experiment:

```bash
python -m traacer.experiments.AECExperiment --list-devices
```

Most output paths are relative to the directory from which the command is run.
The repository's existing reference results are in
`src/traacer/experiments/measurements/`.

## Using trAACer as a library

Communication chains are assembled from asynchronous sources, stages, and
sinks:

- a `Source` produces typed blocks such as `BitBlock` or `AudioSampleBlock`
- a `Stage` transforms an asynchronous stream
- a `Sink` consumes the final stream
- `ProcessorStage` adapts a buffered `StreamingProcessor` into a stage

This small example converts a bit sequence to OOK audio and plays it through
the default output device:

```python
import asyncio

import numpy as np

from traacer.network_stack.layer.base import BitBlock
from traacer.network_stack.layer.device_layer.SDDeviceLayer import SounddeviceSink
from traacer.network_stack.layer.physical_layer.modulation.ask import (
    BitsToOOKAudioSamples,
    OOKConfig,
)


async def bit_source():
    yield BitBlock(
        data=np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=np.uint8),
        is_final=True,
    )


async def main():
    config = OOKConfig(
        sample_rate=48_000,
        symbol_rate=20,
        carrier_frequency=4_000,
        amplitude=0.8,
    )
    audio_stream = BitsToOOKAudioSamples(config).process(bit_source())
    await SounddeviceSink(config.sample_rate).consume(audio_stream)


asyncio.run(main())
```

For a full bidirectional chain, use an experiment such as
`ChannelCodingExercise.py` or `OFDMQAMExercise.py` as a reference. These show
how to add preambles, framing, error correction, browser capture,
demodulation, and result reporting.

## Project layout

```text
trAACer/
├── pyproject.toml
├── Makefile
└── src/traacer/
    ├── experiments/              runnable demonstrations and measurements
    ├── metrics/                  BER comparison and plotting helpers
    ├── network_stack/layer/
    │   ├── base.py               stream, block, source, stage, and sink types
    │   ├── device_layer/         browser, sound-device, and WAV adapters
    │   ├── payload_layer/        application payload helpers
    │   └── physical_layer/       framing, coding, sync, and modulation
    └── receiver/
        ├── server.py             FastAPI/WebSocket receiver service
        ├── static/index.html     browser microphone client
        └── certs/                local development certificate and key
```

## Troubleshooting

**The phone cannot open the receiver page**

- Confirm both devices are on the same LAN and use the printed LAN address,
  not `localhost` or `127.0.0.1`.
- Allow inbound connections to port `8000` in the computer's firewall.
- Stop any previous experiment that is still using the port.

**The browser reports a certificate or microphone error**

- Open the receiver URL directly and accept the local certificate warning.
- Grant microphone permission and press **Manual start** again.
- Use the HTTPS URL printed by the experiment; browsers generally restrict
  microphone capture on insecure pages.

**There is no useful received signal**

- Raise the computer's output volume and move the microphone closer.
- Disable Bluetooth routing if playback or capture uses an unexpected device.
- Avoid clipping: reduce a modulation config's `amplitude` if the waveform is
  visibly flattened.
- Keep browser noise suppression, automatic gain control, and echo
  cancellation disabled; the supplied page requests raw capture settings,
  although device support varies.

**`sounddevice` raises a PortAudio or device error**

```bash
python -m traacer.experiments.AECExperiment --list-devices
```

Select valid devices with `--input-device` and `--output-device`, or configure
the operating system's default audio devices.

**A plot cannot open in a headless environment**

Use `--no-plot` for the BER sweep scripts or `--no-show` for
`AECExperiment`. The result files will still be written.

## Development

The package uses a `src` layout and editable installation, so changes under
`src/traacer` are picked up without reinstalling. There is currently no
automated test suite; a lightweight syntax check is:

```bash
python -m compileall -q src
```

When adding a new pipeline component, preserve block metadata and the
`is_final` flag so buffering, framing, and downstream stages can finish
correctly.
