from typing import Protocol

from traacer.network_stack.packet import LinkLayerPacket, DeviceLayerPacket


class PhysicalLayerSender(Protocol):
    def send_down(self, packet: LinkLayerPacket):
        pass

class PhysicalLayerReceiver(Protocol):
    def send_up(self, packet: DeviceLayerPacket):
        pass