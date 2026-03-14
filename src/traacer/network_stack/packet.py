from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Packet:
    bytes: bytes

class PayloadLayerPacket(Packet):
    pass

class LinkLayerPacket(Packet):
    pass

class PhysicalLayerPacket(Packet):
    pass

class DeviceLayerPacket(Packet):
    pass