from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class Packet:
    bytes: np.ndarray

class PayloadLayerPacket(Packet):
    pass

class LinkLayerPacket(Packet):
    pass

class PhysicalLayerPacket(Packet):
    pass

class DeviceLayerPacket(Packet):
    pass