from typing import Iterable

import numpy as np

from traacer.metrics.bit_comparator import bit_send_queue
from traacer.network_stack.layer.base import Stage, BitBlock, Stream, StreamingProcessor, AudioSampleBlock


class Repeat3(Stage[BitBlock, BitBlock]):
    async def process(self, stream: Stream[BitBlock]) -> Stream[BitBlock]:
        async for block in stream:
            bit_send_queue.put(np.repeat(block.data, 3).astype(block.data.dtype))
            print("Bits sent: " + str(np.repeat(block.data, 3).astype(block.data.dtype)))
            yield BitBlock(
                data=np.repeat(block.data, 3).astype(block.data.dtype),
                is_final=block.is_final,
                metadata=block.metadata,
            )

class Repeat3Corrector(StreamingProcessor[BitBlock, BitBlock]):
    def __init__(self):
        self.buffer = np.empty(0, dtype=np.uint8)

    def push(self, block: BitBlock) -> Iterable[BitBlock]:
        self.buffer = np.concatenate([self.buffer, block.data])

        num_complete_groups = len(self.buffer) // 3
        usable_len = num_complete_groups * 3

        if usable_len == 0:
            self.buffer = np.empty(0, dtype=np.uint8)
            return []

        groups = self.buffer[:usable_len].reshape(-1, 3)
        corrected = (np.sum(groups, axis=1) >= 2).astype(np.uint8)

        self.buffer = np.empty(0, dtype=np.uint8)

        return [
            BitBlock(
                data=corrected,
                is_final=block.is_final,
                metadata=block.metadata,
            )
        ]

class AddGuardSilence(Stage[AudioSampleBlock, AudioSampleBlock]):
    def __init__(self, num_samples: int):
        self.num_samples = num_samples

    async def process(
        self,
        stream: Stream[AudioSampleBlock],
    ) -> Stream[AudioSampleBlock]:
        guard = np.zeros(self.num_samples, dtype=np.float64)

        async for block in stream:
            yield AudioSampleBlock(
                data=np.concatenate([guard, block.data, guard]),
                is_final=block.is_final,
                metadata=block.metadata,
            )

class Hamming74Encode(Stage[BitBlock, BitBlock]):
    async def process(
        self,
        stream: Stream[BitBlock],
    ) -> Stream[BitBlock]:
        async for block in stream:
            bits = np.asarray(block.data, dtype=np.uint8).reshape(-1)

            if np.any((bits != 0) & (bits != 1)):
                raise ValueError("BitBlock data must contain only 0 and 1")

            padding_bits = (-len(bits)) % 4

            if padding_bits:
                bits = np.pad(bits, (0, padding_bits))

            words = bits.reshape(-1, 4)

            d1 = words[:, 0]
            d2 = words[:, 1]
            d3 = words[:, 2]
            d4 = words[:, 3]

            p1 = d1 ^ d2 ^ d4
            p2 = d1 ^ d3 ^ d4
            p3 = d2 ^ d3 ^ d4

            encoded = np.column_stack((
                p1,
                p2,
                d1,
                p3,
                d2,
                d3,
                d4,
            )).reshape(-1)

            metadata = dict(block.metadata)

            print("Hamming bits" + str(encoded))

            yield BitBlock(
                data=encoded,
                is_final=block.is_final,
                metadata=metadata,
            )


class Hamming74Decode(StreamingProcessor[BitBlock, BitBlock]):
    def __init__(self):
        self.buffer = np.empty(0, dtype=np.uint8)

    def push(self, block: BitBlock) -> Iterable[BitBlock]:
        bits = np.asarray(block.data, dtype=np.uint8).reshape(-1)

        if np.any((bits != 0) & (bits != 1)):
            raise ValueError("BitBlock data must contain only 0 and 1")

        self.buffer = np.concatenate([self.buffer, bits])

        num_complete_codewords = len(self.buffer) // 7
        usable_len = num_complete_codewords * 7

        if usable_len == 0:
            if block.is_final and len(self.buffer) != 0:
                raise ValueError(
                    "Final Hamming(7,4) stream contains an incomplete codeword"
                )

            return []

        words = self.buffer[:usable_len].reshape(-1, 7).copy()
        self.buffer = self.buffer[usable_len:]

        s1 = words[:, 0] ^ words[:, 2] ^ words[:, 4] ^ words[:, 6]
        s2 = words[:, 1] ^ words[:, 2] ^ words[:, 5] ^ words[:, 6]
        s3 = words[:, 3] ^ words[:, 4] ^ words[:, 5] ^ words[:, 6]

        error_positions = (
            s1.astype(np.uint8)
            + 2 * s2.astype(np.uint8)
            + 4 * s3.astype(np.uint8)
        )

        error_rows = np.flatnonzero(error_positions)

        if len(error_rows):
            words[
                error_rows,
                error_positions[error_rows] - 1,
            ] ^= 1

        decoded = words[:, [2, 4, 5, 6]].reshape(-1)

        if block.is_final and len(self.buffer) != 0:
            raise ValueError(
                "Final Hamming(7,4) stream contains an incomplete codeword"
            )

        return [
            BitBlock(
                data=decoded,
                is_final=block.is_final,
                metadata=block.metadata,
            )
        ]

