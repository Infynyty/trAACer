import wave
from wave import Wave_write, Wave_read

import numpy as np

from traacer.network_stack.layer.physical_layer.PhysicalLayer import PhysicalLayerReceiver
from traacer.network_stack.packet import PhysicalLayerPacket, DeviceLayerPacket


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

        data = packet.data

        if np.issubdtype(packet.data.dtype, np.floating):
            data = np.clip(data, -1, 1)
            data = (data * 32767).astype(np.int16)

        self.writer.writeframes(data.tobytes())

    def stop(self):
        self.writer.close()

class WAVDeviceLayerReceiver:
    """
    Device Layer implementation that reads the entire WAV file
    and emits a single PhysicalLayerPacket.
    """

    def __init__(self, phy_receiver: PhysicalLayerReceiver, path: str):
        self.phy_receiver: PhysicalLayerReceiver = phy_receiver
        self.path = path
        self.reader: Wave_read | None = None
        self.open = False
        self._consumed = False

    def start(self):
        wf = wave.open(self.path, "rb")

        if wf.getnchannels() != 1:
            raise ValueError("Only mono WAV supported.")
        if wf.getsampwidth() != 2:
            raise ValueError("Only 16-bit WAV supported.")

        self.reader = wf
        self.open = True
        self._consumed = False

    def send_up(self):
        if not self.open:
            raise ValueError("WAV Device Reader not open.")

        if self._consumed:
            return

        n_frames = self.reader.getnframes()
        raw_bytes = self.reader.readframes(n_frames)

        if len(raw_bytes) == 0:
            return

        data = np.frombuffer(raw_bytes, dtype=np.int16)

        data = data.astype(np.float32) / 32767.0

        self._consumed = True

        packet = DeviceLayerPacket(data)
        self.phy_receiver.send_up(packet)

    def stop(self):
        if self.reader:
            self.reader.close()
        self.open = False