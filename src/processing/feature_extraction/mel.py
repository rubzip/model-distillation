import torch
import torchaudio
import torchaudio.transforms as T

from ..base import BaseProcessing


class MelProcessor(BaseProcessing):
    def __init__(
        self, n_mels: int = 128, target_sr: int = 16000, n_fft: int = 400, hop_length: int = 160
    ):
        self.n_mels = n_mels
        self.target_sr = target_sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.mel_transform = T.MelSpectrogram(
            sample_rate=self.target_sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            power=2.0,
        )

    def process(self, waveform: torch.Tensor) -> torch.Tensor:
        """Convierte un waveform en un espectrograma mel."""
        mel_spec = self.mel_transform(waveform)
        mel_spec_db = torchaudio.transforms.AmplitudeToDB()(mel_spec)
        return mel_spec_db

    def to(self, device: torch.device) -> 'MelProcessor':
        self.mel_transform = self.mel_transform.to(device)
        return self
