import torch
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
from torchaudio.functional import resample


class RandomSplit(object):
    def __init__(self, sample_length, nb_samples=5, seed=1992):
        self.sample_length = sample_length
        self.nb_samples = nb_samples
        torch.manual_seed(seed)

    def pad_tensor(self, x, pad_left, pad_right):
        return torch.nn.functional.pad(
            x,
            (pad_left, pad_right),
            mode="constant",
            value=0.0,
        )

    def get_boundaries(self, x):
        x_size = x.size(0)
        if x_size < self.sample_length:
            _min = x_size - self.sample_length
            _max = 0
        else:
            _min = 0
            _max = x_size - self.sample_length
        return _min, _max

    def get_sample(self, x, lower, upper):
        x_size = x.size(0)
        start = torch.randint(lower, upper, (1,))
        end = start + self.sample_length
        if start < 0:
            return self.pad_tensor(x, torch.abs(start), end - x_size)
        return x[start:end]

    def __call__(self, x):
        lower, upper = self.get_boundaries(x)
        res = []
        for i in range(self.nb_samples):
            res.append(self.get_sample(x, lower, upper))
        return torch.stack(res)
    

class Split(object):
    def __init__(self, sample_length, nb_samples=None):
        self.sample_length = sample_length
        self.nb_samples = nb_samples

    def __call__(self, x):
        res = list(torch.split(x, self.sample_length))
        if res[-1].size(0) < self.sample_length:
            res[-1] = torch.nn.functional.pad(
                res[-1],
                (0, self.sample_length - res[-1].size(0)),
                mode="constant",
                value=0.0,
            )
        if self.nb_samples is None:
            return torch.stack(res)
        return torch.stack(res[:self.nb_samples])


class NormalizeMelSpec(object):
    def __init__(self, eps=1e-6, mean=None, std=None):
        self.eps = eps
        self.mean = mean
        self.std = std

    def __call__(self, x):
        mean = self.mean or x.mean()
        std = self.std or x.std()
        x = (x - mean) / (std + self.eps)
        return x


class Preprocessing(torch.nn.Module):
    def __init__(
        self,
        resample_freq=32_000,
        duration=5,
        n_fft=2_048,
        n_mels=128,
        hop_length=512,
        top_db=80,
        nb_samples=5,
        split_method="random"
    ):
        super().__init__()
        self.resample_freq = resample_freq
        sample_length = duration * resample_freq
        assert split_method in ["random", "ordered"]
        if split_method == "random":
            self.split = RandomSplit(sample_length=sample_length, nb_samples=nb_samples)
        elif split_method == "ordered":
            self.split = Split(sample_length=sample_length, nb_samples=nb_samples)
        self.mel_spec = MelSpectrogram(
            sample_rate=resample_freq,
            n_fft=n_fft,
            hop_length=hop_length,
            center=True,
            pad_mode="reflect",
            power=2.0,
            norm="slaney",
            n_mels=n_mels,
            mel_scale="htk",
        )
        self.amp_to_db = AmplitudeToDB(top_db=top_db)
        self.mono_to_color = NormalizeMelSpec()

    def _resample(self, waveform, orig_freq):
        return resample(
            waveform,
            orig_freq,
            self.resample_freq,
            resampling_method="sinc_interp_kaiser",
        )

    def forward(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        waveform = self._resample(waveform, sample_rate)
        waveform = waveform.squeeze()
        waveforms = self.split(waveform)
        images = []
        for waveform in waveforms:
            image = self.mel_spec(waveform)
            image = self.amp_to_db(image)
            image = self.mono_to_color(image)
            images.append(image)
        images = torch.stack(images)
        return images
