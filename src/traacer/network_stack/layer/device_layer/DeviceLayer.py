from typing import Protocol

from traacer.network_stack.packet import PhysicalLayerPacket


class DeviceLayerSender(Protocol):
    def send_down(self, packet: PhysicalLayerPacket):
        pass

class DeviceLayerReceiver(Protocol):
    def send_up(self):
        pass