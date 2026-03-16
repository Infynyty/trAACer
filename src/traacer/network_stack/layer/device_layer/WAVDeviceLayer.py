import wave
from wave import Wave_write

import numpy as np

from traacer.metrics.plot import plot_waveforms
from traacer.network_stack.carriers import sawtooth_wave, sine_wave
from traacer.network_stack.packet import PhysicalLayerPacket


class WAVDeviceLayerSender:
    """
    Device Layer implementation that writes packets into a WAV file.
    """

    def __init__(self, path: str, sample_rate: int = 44100):
        self.path = path
        self.sample_rate = sample_rate
        self.writer: None | Wave_write = None
        self.open = False

    def start(self):
        wf = wave.open(self.path, "wb")
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(self.sample_rate)
        self.writer = wf
        self.open = True

    def send_down(self, packet: PhysicalLayerPacket):
        if not self.open:
            raise ValueError("WAV Device Writer not open.")

        data = packet.bytes

        if np.issubdtype(packet.bytes.dtype, np.floating):
            data = np.clip(data, -1, 1)
            data = (data * 32767).astype(np.int16)

        self.writer.writeframes(data.tobytes())

    def stop(self):
        self.writer.close()