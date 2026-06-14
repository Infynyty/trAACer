import asyncio
import socket
import threading
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import qrcode
import uvicorn

from traacer.metrics.plot import plot_signal, plot_fft, q
from traacer.network_stack.layer.base import StreamingProcessor, BitBlock, DataBlock, IntArray, InT, OutT, Stage, \
    AudioSampleBlock, Stream, CharToBytes, BytesToBits, ProcessorStage, CharBlock, BytesToChar, BitsToBytes
from traacer.network_stack.layer.device_layer.DeviceLayer import DeviceLayerSender
from traacer.network_stack.layer.device_layer.WebserverDeviceLayer import WebserverDeviceLayerReceiverSource, \
    WebserverDeviceLayerSenderSink
from traacer.network_stack.layer.payload_layer.LinkLayer import LinkLayerReceiver
from traacer.network_stack.layer.physical_layer.PhysicalLayerUtil import manchester_decode, bits_to_bytes, \
    bytes_to_bits, manchester_encode
from traacer.network_stack.layer.physical_layer.Synchronization import create_zadoff_chu_preamble, find_preamble_start, \
    PacketAudioSampleBlock, FindPackets, PrependPreamble, create_chirp_preamble
from traacer.network_stack.packet import LinkLayerPacket, PhysicalLayerPacket, DeviceLayerPacket

import matplotlib.pyplot as plt

from traacer.receiver.server import app, cert_path, key_path


class FSKPhysicalLayerSender:
    def __init__(
        self,
        device_layer_sender: DeviceLayerSender,
        amplitude_reduction_factor: float,
        sample_rate: int = 44100,
        freq_zero: float = 1200.0,
        freq_one: float = 2200.0,
        bit_duration: float = 0.05,
        ramp_duration: float = 0.002,
        use_manchester: bool = True,
    ):
        self.sample_rate = sample_rate
        self.freq_zero = freq_zero
        self.freq_one = freq_one
        self.bit_duration = bit_duration
        self.ramp_duration = ramp_duration
        self.use_manchester = use_manchester

        self.amplitude = 1.0 / amplitude_reduction_factor
        self.samples_per_bit = int(self.sample_rate * self.bit_duration)
        self.ramp_samples = min(
            int(self.sample_rate * self.ramp_duration),
            self.samples_per_bit // 2,
        )

        if self.ramp_samples > 0:
            self.ramp = (
                0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, self.ramp_samples))
            ).astype(np.float32)
        else:
            self.ramp = np.empty(0, dtype=np.float32)

        t = np.arange(self.samples_per_bit, dtype=np.float64) / self.sample_rate
        self.symbol_zero = np.sin(2.0 * np.pi * self.freq_zero * t).astype(np.float32)
        self.symbol_one = np.sin(2.0 * np.pi * self.freq_one * t).astype(np.float32)

        self.device_layer_sender = device_layer_sender


    def _apply_ramp(self, symbol: np.ndarray) -> np.ndarray:
        if self.ramp_samples == 0:
            return symbol.copy()

        out = symbol.copy()
        out[:self.ramp_samples] *= self.ramp
        out[-self.ramp_samples:] *= self.ramp[::-1]
        return out

    def send_down(self, packet: LinkLayerPacket):
        bits = np.fromiter(
            bytes_to_bits(packet.data.tobytes()),
            dtype=np.uint8,
        )

        if bits.size == 0:
            self.device_layer_sender.send_down(
                PhysicalLayerPacket(np.empty(0, dtype=np.float32))
            )
            return

        if self.use_manchester:
            bits = manchester_encode(bits)

        zero_symbol = self._apply_ramp(self.amplitude * self.symbol_zero)
        one_symbol = self._apply_ramp(self.amplitude * self.symbol_one)

        symbols = [one_symbol if bit else zero_symbol for bit in bits]
        signal = np.concatenate(symbols).astype(np.float32)
        signal = np.concat([create_zadoff_chu_preamble(), signal])

        q.put(signal)
        self.device_layer_sender.send_down(PhysicalLayerPacket(signal))


class FSKPhysicalLayerReceiver:
    def __init__(
        self,
        link_layer_receiver: LinkLayerReceiver,
        sample_rate: int = 44100,
        freq_zero: float = 1200.0,
        freq_one: float = 2200.0,
        bit_duration: float = 0.05,
        use_manchester: bool = True,
    ):
        self.sample_rate = sample_rate
        self.freq_zero = freq_zero
        self.freq_one = freq_one
        self.bit_duration = bit_duration
        self.use_manchester = use_manchester
        self.samples_per_bit = int(self.sample_rate * self.bit_duration)

        t = np.arange(self.samples_per_bit, dtype=np.float64) / self.sample_rate

        ref_zero_i = np.cos(2.0 * np.pi * self.freq_zero * t)
        ref_zero_q = np.sin(2.0 * np.pi * self.freq_zero * t)
        ref_one_i = np.cos(2.0 * np.pi * self.freq_one * t)
        ref_one_q = np.sin(2.0 * np.pi * self.freq_one * t)

        self.ref_zero_i = ref_zero_i.astype(np.float32)
        self.ref_zero_q = ref_zero_q.astype(np.float32)
        self.ref_one_i = ref_one_i.astype(np.float32)
        self.ref_one_q = ref_one_q.astype(np.float32)

        self.link_layer_receiver = link_layer_receiver


    def _tone_energy(self, symbol: np.ndarray, ref_i: np.ndarray, ref_q: np.ndarray) -> float:
        i = np.dot(symbol, ref_i)
        q = np.dot(symbol, ref_q)
        return float(i * i + q * q)

    def _detect_bits(self, signal: np.ndarray) -> list[int]:
        n_bits = len(signal) // self.samples_per_bit
        if n_bits == 0:
            return []

        bits = []

        for i in range(n_bits):
            s = i * self.samples_per_bit
            e = s + self.samples_per_bit
            symbol = signal[s:e]

            energy_zero = self._tone_energy(symbol, self.ref_zero_i, self.ref_zero_q)
            energy_one = self._tone_energy(symbol, self.ref_one_i, self.ref_one_q)

            bits.append(1 if energy_one > energy_zero else 0)

        return bits

    def send_up(self, packet: DeviceLayerPacket):
        signal = np.asarray(packet.data, dtype=np.float32)
        preamble = create_zadoff_chu_preamble()
        signal_start_index = find_preamble_start(signal, preamble) + len(preamble)
        signal = signal[signal_start_index:]

        detected_bits = self._detect_bits(signal)

        if self.use_manchester:
            detected_bits = manchester_decode(detected_bits)

        data = bits_to_bytes(detected_bits)

        plot_signal(signal)
        plot_fft(signal)
        print(f"Decoded bits: {detected_bits}")
        q.put(signal)
        self.link_layer_receiver.send_up(
            PhysicalLayerPacket(np.frombuffer(data, dtype=np.uint8))
        )

@dataclass(frozen=True, slots=True)
class FSKSymbolBlock(DataBlock[IntArray]):
    pass

@dataclass(frozen=True, slots=True)
class FSKConfig:
    sample_rate: int
    symbol_rate: float

    frequencies: np.typing.NDArray[np.float64]

    amplitude: float = 1.0
    continuous_phase: bool = True

    @property
    def samples_per_symbol(self) -> int:
        return round(self.sample_rate / self.symbol_rate)

    @property
    def order(self) -> int:
        return len(self.frequencies)

    @property
    def bits_per_symbol(self) -> int:
        value = np.log2(self.order)

        if not value.is_integer():
            raise ValueError("FSK order must be a power of two to map bits directly to symbols.")

        return int(value)

    @classmethod
    def from_spacing(
        cls,
        sample_rate: int,
        symbol_rate: float,
        order: int,
        center_frequency: float,
        frequency_spacing: float,
        amplitude: float = 1.0,
        continuous_phase: bool = True,
    ) -> "FSKConfig":
        if order < 2:
            raise ValueError("FSK order must be at least 2.")

        offsets = np.arange(order, dtype=np.float64) - (order - 1) / 2
        frequencies = center_frequency + offsets * frequency_spacing

        return cls(
            sample_rate=sample_rate,
            symbol_rate=symbol_rate,
            frequencies=frequencies,
            amplitude=amplitude,
            continuous_phase=continuous_phase,
        )

class BitsToFSKSymbols(StreamingProcessor[BitBlock, FSKSymbolBlock]):

    def __init__(self, config: FSKConfig):
        self.config = config
        self.buffer = np.empty(0, dtype=np.uint64)

    def push(self, block: BitBlock) -> Iterable[FSKSymbolBlock]:
        bits = np.concatenate([self.buffer, block.data])

        bits_per_symbol = self.config.bits_per_symbol
        usable_len = len(bits) - (len(bits) % bits_per_symbol)

        if block.is_final and len(bits) % bits_per_symbol:
            pad_len = bits_per_symbol - (len(bits) % bits_per_symbol)
            bits = np.pad(bits, (0, pad_len), constant_values=0)
            usable_len = len(bits)

        usable = bits[:usable_len]
        self.buffer = bits[usable_len:]

        if len(usable) == 0:
            return []

        grouped = usable.reshape(-1, bits_per_symbol)

        weights = 1 << np.arange(bits_per_symbol - 1, -1, -1, dtype=np.uint64)
        symbols = grouped @ weights
        symbols = symbols.astype(np.uint64)

        return [
            FSKSymbolBlock(
                data=symbols,
                is_final=block.is_final,
                metadata=block.metadata,
            )
        ]

class FSKSymbolsToAudioSamples(Stage[FSKSymbolBlock, AudioSampleBlock]):
    def __init__(self, config: FSKConfig):
        self.config = config
        self.phase = 0.0

    async def process(self, stream: Stream[FSKSymbolBlock]) -> Stream[AudioSampleBlock]:
        samples_per_symbol = self.config.samples_per_symbol
        sample_rate = self.config.sample_rate
        amplitude = self.config.amplitude

        async for block in stream:
            symbols = block.data.astype(np.intp)

            if np.any(symbols >= self.config.order):
                raise ValueError("FSK symbol index out of range.")

            if np.any(symbols < 0):
                raise ValueError("FSK symbol index out of range.")

            chunks: list[np.typing.NDArray[np.float64]] = []

            for symbol in symbols:
                frequency = self.config.frequencies[symbol]

                n = np.arange(samples_per_symbol, dtype=np.float64)

                if self.config.continuous_phase:
                    phase = self.phase + 2 * np.pi * frequency * n / sample_rate
                    self.phase = float(
                        (phase[-1] + 2 * np.pi * frequency / sample_rate) % (2 * np.pi)
                    )
                else:
                    phase = 2 * np.pi * frequency * n / sample_rate

                chunks.append(amplitude * np.sin(phase))

            if not chunks:
                continue

            samples = np.concatenate(chunks).astype(np.float64)

            yield AudioSampleBlock(
                data=samples,
                is_final=block.is_final,
                metadata=block.metadata,
            )

class PacketAudioSamplesToFSKSymbols(
    StreamingProcessor[PacketAudioSampleBlock, FSKSymbolBlock]
):
    def __init__(self, config: FSKConfig):
        self.config = config
        self.buffer = np.empty(0, dtype=np.float64)

        n = np.arange(config.samples_per_symbol, dtype=np.float64)
        phases = 2 * np.pi * config.frequencies[:, None] * n[None, :] / config.sample_rate

        self.sin_refs = np.sin(phases)
        self.cos_refs = np.cos(phases)

    def push(self, block: PacketAudioSampleBlock) -> Iterable[FSKSymbolBlock]:
        samples = np.concatenate([self.buffer, block.data.astype(np.float64)])

        samples_per_symbol = self.config.samples_per_symbol
        usable_len = len(samples) - (len(samples) % samples_per_symbol)

        usable = samples[:usable_len]
        self.buffer = samples[usable_len:]

        if block.is_final and len(self.buffer):
            self.buffer = np.empty(0, dtype=np.float64)

        if len(usable) == 0:
            return []

        symbol_windows = usable.reshape(-1, samples_per_symbol)

        i_corr = symbol_windows @ self.cos_refs.T
        q_corr = symbol_windows @ self.sin_refs.T

        energy = i_corr * i_corr + q_corr * q_corr
        symbols = np.argmax(energy, axis=1).astype(np.uint64)

        return [
            FSKSymbolBlock(
                data=symbols,
                is_final=block.is_final,
                metadata=block.metadata,
            )
        ]

class FSKSymbolsToBits(StreamingProcessor[FSKSymbolBlock, BitBlock]):
    def __init__(self, config: FSKConfig):
        self.config = config

    def push(self, block: FSKSymbolBlock) -> Iterable[BitBlock]:
        symbols = block.data.astype(np.uint64)

        if np.any(symbols >= self.config.order):
            raise ValueError("FSK symbol index out of range.")

        bits_per_symbol = self.config.bits_per_symbol

        shifts = np.arange(bits_per_symbol - 1, -1, -1, dtype=np.uint64)
        bits = ((symbols[:, None] >> shifts) & 1).astype(np.uint64)
        bits = bits.reshape(-1)

        print("Bits received: " + str(bits))

        return [
            BitBlock(
                data=bits,
                is_final=block.is_final,
                metadata=block.metadata,
            )
        ]

async def string_source(messages: list[str]) -> Stream[CharBlock]:
    for message in messages:
        print("Message: " + message)
        yield CharBlock(data=message, is_final=True)

async def user_input_source() -> Stream[CharBlock]:
    while True:
        text = await asyncio.to_thread(input, "Input some text...")
        yield CharBlock(data=text, is_final=True)


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


