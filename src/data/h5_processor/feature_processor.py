import torch
from .base_processor import AbstractH5BatchProcessor

class H5FeatureProcessor(AbstractH5BatchProcessor):
    def _process_batch(self, batch_tensor: torch.Tensor) -> torch.Tensor:
        # Los extractores de torchaudio suelen esperar [Batch, Time] o [Batch, Channels, Time]
        # Aplicamos la transformación directamente
        features = self.processor(batch_tensor)
        return features
