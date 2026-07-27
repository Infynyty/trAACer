import asyncio
import threading

import numpy as np

from traacer.network_stack.layer.base import (
    RunningSignalPlot,
    repeated_bit_source,
    run_webserver,
)
from traacer.network_stack.layer.device_layer.WebserverDeviceLayer import (
    WebserverDeviceLayerReceiverSource,
    WebserverDeviceLayerSenderSink,
)
from traacer.network_stack.layer.physical_layer.modulation.ask import (
    ASKConfig,
    BitsToASKAudioSamples,
)
from traacer.receiver.server import cert_path


BIT_SEQUENCE = np.array([0, 0, 0, 1, 0, 1, 1, 1], dtype=np.uint8)
PLOT_WINDOW_SECONDS = 1.0
PLOT_UPDATE_INTERVAL_SECONDS = 0.1


async def main() -> None:
    webserver_thread = threading.Thread(target=run_webserver, daemon=True)
    webserver_thread.start()

    ask_config = ASKConfig(
        sample_rate=48_000,
        symbol_rate=5,
        carrier_frequency=4_000,
        order=4,
        continuous_phase=True,
    )
    if len(BIT_SEQUENCE) % ask_config.bits_per_symbol:
        raise ValueError("BIT_SEQUENCE must contain complete ASK symbols")

    bit_stream = repeated_bit_source(BIT_SEQUENCE, delay_seconds=0.0)
    audio_stream = BitsToASKAudioSamples(ask_config).process(bit_stream)

    sender = WebserverDeviceLayerSenderSink(
        server_url="https://127.0.0.1:8000",
        sample_rate=ask_config.sample_rate,
        cert_path=cert_path,
    )
    receiver = WebserverDeviceLayerReceiverSource(
        target_sample_rate=ask_config.sample_rate,
        cert_path=cert_path,
    )
    plot = RunningSignalPlot(
        sample_rate=ask_config.sample_rate,
        window_seconds=PLOT_WINDOW_SECONDS,
        update_interval_seconds=PLOT_UPDATE_INTERVAL_SECONDS,
        title="Received ASK signal",
        y_limit=0.05
    )

    async def plot_received_audio() -> None:
        async for _ in plot.process(receiver.stream()):
            pass

    await asyncio.gather(sender.consume(audio_stream), plot_received_audio())


if __name__ == "__main__":
    asyncio.run(main())
