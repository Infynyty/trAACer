from traacer.network_stack.packet import PhysicalLayerPacket
import sounddevice as sd


class SDDeviceLayerSender:

    def __init__(self, sample_rate: int):
        self.sample_rate = sample_rate

    def send_down(self, packet: PhysicalLayerPacket):
        sd.play(packet.bytes, samplerate=self.sample_rate)
        sd.wait()