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