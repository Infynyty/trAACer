from typing import Protocol, List

from traacer.network_stack.packet import PayloadLayerPacket, PhysicalLayerPacket


class LinkLayerReceiver(Protocol):
    def send_up(self, packet: PhysicalLayerPacket):
        pass

class LinkLayerSender(Protocol):
    def send_down(self, packet: PayloadLayerPacket):
        pass


class NoopLinkLayerReceiver:

    def __init__(self):
        self.packets = []

    def send_up(self, packet: PhysicalLayerPacket):
        self.packets.append(packet)
