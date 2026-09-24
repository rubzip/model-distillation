import torch
from .base_processor import AbstractH5BatchProcessor

class H5EmbeddingProcessor(AbstractH5BatchProcessor):
    def _process_batch(self, batch_tensor: torch.Tensor) -> torch.Tensor:
        if batch_tensor.dim() == 2:
            batch_tensor = batch_tensor.unsqueeze(1) # Añade canal
            
        embeddings = self.processor(batch_tensor)
        
        if embeddings.dim() > 2:
            embeddings = embeddings.squeeze()
            
        return embeddings
