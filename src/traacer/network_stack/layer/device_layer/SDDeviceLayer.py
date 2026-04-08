import sounddevice as sd

from traacer.network_stack.packet import PhysicalLayerPacket


class SDDeviceLayerSender:

    def __init__(self, sample_rate: int):
        self.sample_rate = sample_rate

    def send_down(self, packet: PhysicalLayerPacket):
        sd.play(packet.data, samplerate=self.sample_rate)
        sd.wait()