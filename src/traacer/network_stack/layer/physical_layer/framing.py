from typing import Iterable

import numpy as np

from traacer.network_stack.layer.base import Stage, BitBlock, Stream, StreamingProcessor


class AddBitLengthHeader(Stage[BitBlock, BitBlock]):
    def __init__(self, header_bits: int):
        self.header_bits = header_bits

    async def process(self, stream: Stream[BitBlock]) -> Stream[BitBlock]:
        async for block in stream:
            payload_len = len(block.data)

            if payload_len >= 2 ** self.header_bits:
                raise ValueError(
                    f"Payload too large for {self.header_bits}-bit length header: "
                    f"{payload_len} bits"
                )

            header = int_to_bits(payload_len, self.header_bits)

            yield BitBlock(
                data=np.concatenate([header, block.data.astype(np.uint8)]),
                is_final=block.is_final,
                metadata=block.metadata,
            )

class RemoveBitLengthHeader(StreamingProcessor[BitBlock, BitBlock]):
    def __init__(self, header_bits: int):
        self.header_bits = header_bits
        self.buffer = np.empty(0, dtype=np.uint8)
        self.payload_len: int | None = None

    def push(self, block: BitBlock) -> Iterable[BitBlock]:
        self.buffer = np.concatenate([block.data.astype(np.uint8)]) #todo fix

        if self.payload_len is None:
            if len(self.buffer) < self.header_bits:
                self.buffer = np.empty(0, dtype=np.uint8)
                return []

            self.payload_len = bits_to_int(self.buffer[:self.header_bits])
            print("Payload length is " + str(self.payload_len))
            self.buffer = self.buffer[self.header_bits:]

        if len(self.buffer) < self.payload_len:
            self.buffer = np.empty(0, dtype=np.uint8)
            self.payload_len = None
            return []

        payload = self.buffer[:self.payload_len]
        self.buffer = np.empty(0, dtype=np.uint8)

        print("Final bits received: " + str(payload))

        out = BitBlock(
            data=payload,
            is_final=block.is_final,
            metadata=block.metadata,
        )

        self.payload_len = None

        return [out]

def int_to_bits(value: int, width: int) -> np.ndarray:
    return np.array(
        [(value >> shift) & 1 for shift in range(width - 1, -1, -1)],
        dtype=np.uint8,
    )


def bits_to_int(bits: np.ndarray) -> int:
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value