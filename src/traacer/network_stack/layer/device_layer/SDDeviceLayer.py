import sounddevice as sd

from traacer.network_stack.layer.base import Stream, AudioSampleBlock, Sink

class SounddeviceSink(Sink[AudioSampleBlock]):
    """
    Device layer sink that plays incoming audio sample blocks using sounddevice.
    """

    def __init__(self, sample_rate: int):
        self.sample_rate = sample_rate

    async def consume(self, stream: Stream[AudioSampleBlock]) -> None:
        async for packet in stream:
            sd.play(packet.data, samplerate=self.sample_rate)
            sd.wait()
