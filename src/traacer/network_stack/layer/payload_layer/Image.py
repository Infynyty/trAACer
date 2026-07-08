import math

from dataclasses import dataclass

import asyncio

import numpy as np
from matplotlib import pyplot as plt

from traacer.network_stack.layer.base import Stream, ByteBlock, Sink, DataBlock, Source
from IPython.display import clear_output


@dataclass(frozen=True, slots=True)
class ImageConfig:
    width: int = 64
    height: int = 64
    packet_size: int = 16
    index_size: int = 2
    fill_value: int = 128

    @property
    def num_pixels(self) -> int:
        return self.width * self.height

    @property
    def payload_size(self) -> int:
        return self.packet_size - self.index_size

    @property
    def num_chunks(self) -> int:
        return math.ceil(self.num_pixels / self.payload_size)


class TestImageSource(Source[ByteBlock]):
    def __init__(self, config: ImageConfig):
        self.config = config

    async def stream(self) -> Stream[ByteBlock]:
        image = self.create_test_image()
        image_bytes = image.ravel()

        for chunk_index in range(self.config.num_chunks):
            start = chunk_index * self.config.payload_size
            end = min(start + self.config.payload_size, self.config.num_pixels)

            payload = np.full(
                self.config.payload_size,
                self.config.fill_value,
                dtype=np.uint8,
            )
            payload[: end - start] = image_bytes[start:end]

            packet = np.empty(self.config.packet_size, dtype=np.uint8)
            packet[0] = (chunk_index >> 8) & 0xFF
            packet[1] = chunk_index & 0xFF
            packet[2:] = payload

            yield ByteBlock(
                data=packet,
                is_final=chunk_index == self.config.num_chunks - 1,
            )

    def create_test_image(self) -> np.ndarray:
        image = np.zeros(
            (self.config.height, self.config.width),
            dtype=np.uint8,
        )

        image[:, : self.config.width // 4] = 40
        image[:, self.config.width // 4 : self.config.width // 2] = 100
        image[:, self.config.width // 2 : 3 * self.config.width // 4] = 180
        image[:, 3 * self.config.width // 4 :] = 240

        image[
            self.config.height // 4 : 3 * self.config.height // 4,
            self.config.width // 4 : 3 * self.config.width // 4,
        ] = 255

        image[
            3 * self.config.height // 8 : 5 * self.config.height // 8,
            3 * self.config.width // 8 : 5 * self.config.width // 8,
        ] = 0

        return image

    def display_image(self) -> None:
        image = self.create_test_image()

        plt.figure(figsize=(6, 6))
        plt.imshow(image, cmap="gray", vmin=0, vmax=255)
        plt.axis("off")
        plt.show()


class ImageDisplaySink(Sink[ByteBlock]):
    def __init__(self, config: ImageConfig):
        self.config = config
        self.image_bytes = np.full(
            config.num_pixels,
            config.fill_value,
            dtype=np.uint8,
        )
        self.received_chunks: set[int] = set()
        self.invalid_packets = 0

    async def consume(self, stream: Stream[ByteBlock]) -> None:
        async for block in stream:
            self.write_packet(block.data)
            self.show()

            if block.is_final:
                break

    def write_packet(self, packet: np.ndarray) -> None:
        packet = np.asarray(packet, dtype=np.uint8)

        if len(packet) != self.config.packet_size:
            self.invalid_packets += 1
            return

        chunk_index = (int(packet[0]) << 8) | int(packet[1])

        if chunk_index >= self.config.num_chunks:
            self.invalid_packets += 1
            return

        payload = packet[2:]

        start = chunk_index * self.config.payload_size
        end = min(start + len(payload), self.config.num_pixels)
        length = end - start

        if length <= 0:
            self.invalid_packets += 1
            return

        self.image_bytes[start:end] = payload[:length]
        self.received_chunks.add(chunk_index)

    def show(self) -> None:
        image = self.image_bytes.reshape(
            self.config.height,
            self.config.width,
        )

        clear_output()
        plt.figure(figsize=(5, 5))
        plt.imshow(image, cmap="gray", vmin=0, vmax=255)
        plt.title(
            f"Received {len(self.received_chunks)} / {self.config.num_chunks} chunks, "
            f"invalid packets: {self.invalid_packets}"
        )
        plt.axis("off")
        plt.show()
