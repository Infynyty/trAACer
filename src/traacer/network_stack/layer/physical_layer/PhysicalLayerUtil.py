import numpy as np


def bytes_to_bits(data: bytes):
    for byte in data:
        for i in range(8):
            yield (byte >> (7 - i)) & 1

def bits_to_bytes(bits: list[int]) -> bytes:
    n_bytes = len(bits) // 8
    out = bytearray()

    for i in range(n_bytes):
        byte = 0
        for bit in bits[i * 8:(i + 1) * 8]:
            byte = (byte << 1) | bit
        out.append(byte)

    return bytes(out)

def manchester_encode(bits: np.ndarray) -> np.ndarray:
    encoded = np.empty(bits.size * 2, dtype=np.uint8)
    encoded[0::2] = bits
    encoded[1::2] = 1 - bits
    return encoded

def manchester_decode(bits: list[int]) -> list[int]:
    n_pairs = len(bits) // 2
    decoded = []

    for i in range(n_pairs):
        first = bits[2 * i]
        second = bits[2 * i + 1]

        if first == 0 and second == 1:
            decoded.append(0)
        elif first == 1 and second == 0:
            decoded.append(1)

    return decoded

