import asyncio
import socket
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass, field
from typing import Generic, TypeAlias, TypeVar

import numpy as np
import qrcode
import uvicorn

from traacer.receiver.server import cert_path, key_path, app

T = TypeVar("T")
InT = TypeVar("InT")
OutT = TypeVar("OutT")

Stream: TypeAlias = AsyncIterator[T]

class Stage(ABC, Generic[InT, OutT]):
    @abstractmethod
    def process(self, stream: Stream[InT]) -> Stream[OutT]:
        """
        Abstract method representing a processing operation on an input stream.

        This method is designed to be implemented by subclasses, where the processing
        logic for transforming the input stream will be defined. The method takes an
        input stream of type InT and produces an output stream of type OutT.

        Args:
            stream (Stream[InT]): The input stream to be processed. The function must be able to handle input with too
            little data for the next out block and input with enough data for multiple out blocks. Left-over data must
            be buffered.

        Returns:
            Stream[OutT]: The processed output stream, with its items represented by
            the template parameter OutT.
        """
        raise NotImplementedError


class Source(ABC, Generic[OutT]):
    @abstractmethod
    def stream(self) -> Stream[OutT]:
        raise NotImplementedError


class Sink(ABC, Generic[InT]):
    @abstractmethod
    async def consume(self, stream: Stream[InT]) -> None:
        raise NotImplementedError


class StreamingProcessor(ABC, Generic[InT, OutT]):
    @abstractmethod
    def push(self, block: InT) -> Iterable[OutT]:
        raise NotImplementedError

    def flush(self) -> Iterable[OutT]:
        return ()


class ProcessorStage(Stage[InT, OutT]):
    def __init__(self, processor: StreamingProcessor[InT, OutT]):
        self.processor = processor

    async def process(self, stream: Stream[InT]) -> Stream[OutT]:
            async for block in stream:
                for out in self.processor.push(block):
                    yield out

            for out in self.processor.flush():
                yield out


@dataclass(frozen=True, slots=True)
class DataBlock(Generic[T]):
    data: T
    is_final: bool = False
    metadata: dict[str, object] = field(default_factory=dict)

BitArray: TypeAlias = np.typing.NDArray[np.uint8]
ByteArray: TypeAlias = np.typing.NDArray[np.uint8]
IntArray: TypeAlias = np.typing.NDArray[np.uint64]
ComplexArray: TypeAlias = np.typing.NDArray[np.complex128]
AudioSampleArray: TypeAlias = np.typing.NDArray[np.float64]

@dataclass(frozen=True, slots=True)
class BitBlock(DataBlock[BitArray]):
    pass

@dataclass(frozen=True, slots=True)
class ByteBlock(DataBlock[ByteArray]):
    pass

@dataclass(frozen=True, slots=True)
class CharBlock(DataBlock[str]):
    pass

@dataclass(frozen=True, slots=True)
class AudioSampleBlock(DataBlock[AudioSampleArray]):
    pass

class CharToBytes(Stage[CharBlock, ByteBlock]):
    async def process(self, stream: Stream[CharBlock]) -> Stream[ByteBlock]:
        async for block in stream:
            data = np.frombuffer(block.data.encode("utf-8"), dtype=np.uint8)

            yield ByteBlock(
                data=data,
                is_final=block.is_final,
                metadata=block.metadata,
            )

class BytesToBits(Stage[ByteBlock, BitBlock]):
    async def process(self, stream: Stream[ByteBlock]) -> Stream[BitBlock]:
        async for block in stream:
            print("Char bits sent: " + str(np.unpackbits(block.data)))
            yield BitBlock(
                data=np.unpackbits(block.data),
                is_final=block.is_final,
                metadata=block.metadata,
            )

class BitsToBytes(StreamingProcessor[BitBlock, ByteBlock]):
    def __init__(self):
        self.buffer: BitArray = np.empty(0, dtype=np.uint8)

    def push(self, block: BitBlock) -> Iterable[ByteBlock]:
        #bits = np.concatenate([self.buffer, block.data])
        bits = block.data
        usable_len = len(bits) - (len(bits) % 8)

        if block.is_final and len(bits) % 8:
            pad_len = 8 - (len(bits) % 8)
            bits = np.pad(bits, (0, pad_len), constant_values=0)
            usable_len = len(bits)

        usable = bits[:usable_len]
        self.buffer: BitArray = np.empty(0, dtype=np.uint8)

        if len(usable) == 0:
            return []

        bytes_ = np.packbits(usable, bitorder="big")
        print("Char Bits received: " + str(bits))

        return [
            ByteBlock(
                data=bytes_,
                is_final=block.is_final,
                metadata=block.metadata,
            )
        ]

    def flush(self) -> Iterable[OutT]:
        return ()

class BytesToChar(Stage[ByteBlock, CharBlock]):
    async def process(self, stream: Stream[ByteBlock]) -> Stream[CharBlock]:
        async for block in stream:
            yield CharBlock(
                data=block.data.tobytes().decode("utf-8", errors="replace"),
                is_final=block.is_final,
                metadata=block.metadata,
            )


async def string_source(text: str) -> Stream[CharBlock]:
    yield CharBlock(data=text, is_final=True)

async def user_input_source() -> Stream[CharBlock]:
    while True:
        text = await asyncio.to_thread(input, "Input some text...")

        yield CharBlock(data=text, is_final=True)

class PrintCharSink(Sink[CharBlock]):
    async def consume(self, stream: Stream[CharBlock]) -> None:
        async for block in stream:
            print(block.data, end="", flush=True)

async def repeated_char_source(
    char: str = "a",
    delay_seconds: float = 1.0,
) -> Stream[CharBlock]:
    if len(char) != 1:
        raise ValueError("char must be exactly one character")

    while True:
        yield CharBlock(data=char, is_final=True)
        await asyncio.sleep(delay_seconds)

async def repeated_bit_source(
    bits,
    delay_seconds: float = 1.0,
) -> Stream[BitBlock]:

    while True:
        yield BitBlock(data=np.asarray(bits, dtype=np.uint8))
        await asyncio.sleep(delay_seconds)


def run_webserver() -> None:
    try:
        def get_lan_ip() -> str:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            try:
                sock.connect(("8.8.8.8", 80))
                return sock.getsockname()[0]
            finally:
                sock.close()

        def print_qr_to_console(url: str) -> None:
            qr = qrcode.QRCode(border=1)
            qr.add_data(url)
            qr.make(fit=True)

            print()
            print(f"Receiver URL: {url}")
            print()

            for row in qr.get_matrix():
                print("".join("██" if cell else "  " for cell in row))

            print()

        host = "0.0.0.0"
        port = 8000
        scheme = "https"

        url = f"{scheme}://{get_lan_ip()}:{port}"
        print_qr_to_console(url)

        config = uvicorn.Config(
            app,
            host=host,
            port=port,
            reload=False,
            ssl_certfile=str(cert_path),
            ssl_keyfile=str(key_path),
            log_level="critical",
        )

        server = uvicorn.Server(config)
        server.run()

    except BaseException as exc:
        print("WEBSERVER THREAD CRASHED:", repr(exc))
        raise