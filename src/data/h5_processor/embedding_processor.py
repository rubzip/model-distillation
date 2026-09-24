from pathlib import Path
from typing import Callable, Union
import torch
from .base_processor import AbstractH5BatchProcessor

class H5EmbeddingProcessor(AbstractH5BatchProcessor):
    def __init__(
        self,
        input_h5_path: Union[Path, str],
        output_h5_path: Union[Path, str],
        processor: Union[Callable, torch.nn.Module],
        input_dataset_name: str = "audio_segments",
        output_dataset_name: str = "data",
        batch_size: int = 512,
        device: torch.device = torch.device("cpu"),
        add_channel_dim: bool = True
    ):
        super().__init__(
            input_h5_path=input_h5_path,
            output_h5_path=output_h5_path,
            processor=processor,
            input_dataset_name=input_dataset_name,
            output_dataset_name=output_dataset_name,
            batch_size=batch_size,
            device=device
        )
        self.add_channel_dim = add_channel_dim

    def _process_batch(self, batch_tensor: torch.Tensor) -> torch.Tensor:
        if self.add_channel_dim and batch_tensor.dim() == 2:
            batch_tensor = batch_tensor.unsqueeze(1) # Añade canal [Batch, 1, Time]
            
        embeddings = self.processor(batch_tensor)
        
        # Aplanar dimensiones sobrantes preservando SIEMPRE la dimensión de batch (dim 0)
        if embeddings.dim() > 2:
            embeddings = torch.flatten(embeddings, start_dim=1)
            
        return embeddings
