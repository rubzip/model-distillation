import torch
import torchaudio


class Wav2VecProcessor:
    def __init__(self, input_sr: int):
        self.bundle = torchaudio.pipelines.WAV2VEC2_BASE
        self.model = self.bundle.get_model()

        if input_sr != self.bundle.sample_rate:
            raise ValueError(
                f"Input sample rate ({input_sr}) does not match the model's expected sample rate ({self.bundle.sample_rate}). Please resample the audio before processing.")
        
        self.model.eval()
    
    def to(self, device: torch.device) -> None:
        self.model = self.model.to(device)
        self.device = device

    def process(self, waveform: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            features, _ = self.model.extract_features(waveform.to(self.device))
            last_layer_features = features[-1]
            embedding = last_layer_features.mean(dim=1).squeeze(0)
        return embedding
