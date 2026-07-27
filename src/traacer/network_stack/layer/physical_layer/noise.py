import numpy as np

from traacer.network_stack.layer.base import Stage, AudioSampleBlock, Stream


class AddAWGN(Stage[AudioSampleBlock, AudioSampleBlock]):
    def __init__(
        self,
        snr_db: float,
        seed: int | None = None,
    ):
        self.snr_db = snr_db
        self.rng = np.random.default_rng(seed)

    async def process(
        self,
        stream: Stream[AudioSampleBlock],
    ) -> Stream[AudioSampleBlock]:
        async for block in stream:
            samples = np.asarray(block.data)
            signal_power = np.mean(np.square(samples.astype(np.float64)))

            if signal_power == 0:
                noisy_samples = samples.copy()
            else:
                snr_linear = 10.0 ** (self.snr_db / 10.0)
                noise_power = signal_power / snr_linear
                noise = self.rng.normal(
                    loc=0.0,
                    scale=np.sqrt(noise_power),
                    size=samples.shape,
                )

                noisy_samples = (
                    samples.astype(np.float64) + noise
                ).astype(samples.dtype)

            yield AudioSampleBlock(
                data=noisy_samples,
                is_final=block.is_final,
                metadata=block.metadata,
            )