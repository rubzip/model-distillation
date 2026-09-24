import joblib
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.decomposition import IncrementalPCA
from pathlib import Path
from typing import Union

from .base_processor import AbstractH5BatchProcessor

class PCAEmbeddingReducer:
    def __init__(self, target_dim: int = 32):
        self.target_dim = target_dim
        self.pca = IncrementalPCA(n_components=target_dim)
        self.is_fitted = False

    def fit(self, train_loader: DataLoader, embedding_index: int = 0) -> None:        
        print(f"Ajustando PCA (Target Dim: {self.target_dim})...")
        for batch in train_loader:
            embeddings = batch[embedding_index]            
            if isinstance(embeddings, torch.Tensor):
                embeddings = embeddings.cpu().numpy()                
            self.pca.partial_fit(embeddings)
            
        self.is_fitted = True
        print(f"PCA Ajustado. Varianza retenida: {sum(self.get_retained_variance()):.4f}")

    def get_retained_variance(self) -> np.ndarray:            
        return self.pca.explained_variance_ratio_

    def transform(self, e: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        if not self.is_fitted:
            raise ValueError("El PCA no está entrenado. Llama a fit() o load() primero.")
            
        is_tensor = isinstance(e, torch.Tensor)
        device = e.device if is_tensor else None
        
        # Convertir a numpy para sklearn
        e_np = e.cpu().numpy() if is_tensor else e
            
        reduced_np = self.pca.transform(e_np)
        
        # Devolver en el mismo formato que entró
        if is_tensor:
            return torch.from_numpy(reduced_np).to(device)
        return reduced_np

class H5SizeReductionProcessor(AbstractH5BatchProcessor):
    def __init__(
        self,
        input_h5_path: Union[Path, str],
        output_h5_path: Union[Path, str],
        reducer: PCAEmbeddingReducer,  # Inyectamos la dependencia
        input_dataset_name: str = "embeddings",
        output_dataset_name: str = "data",
        batch_size: int = 1024,
        device: torch.device = torch.device("cpu")
    ):
        # Validación temprana: Protegemos el pipeline asegurando que el modelo esté listo
        if not reducer.is_fitted:
            raise ValueError("El reductor debe estar entrenado (is_fitted=True) antes de procesar el HDF5.")

        super().__init__(
            input_h5_path=input_h5_path,
            output_h5_path=output_h5_path,
            processor=reducer, 
            input_dataset_name=input_dataset_name,
            output_dataset_name=output_dataset_name,
            batch_size=batch_size,
            device=device
        )

    def _process_batch(self, batch_tensor: torch.Tensor) -> torch.Tensor:
        # Validación de integridad de los datos
        if batch_tensor.dim() != 2:
            raise ValueError(
                f"El reductor PCA espera tensores 2D [Batch, Embedding_Dim], "
                f"pero recibió un tensor {batch_tensor.dim()}D."
            )
            
        # Como nuestro reductor ahora maneja tensores nativamente en su método transform(),
        # la delegación es directa y limpia.
        reduced_tensor = self.processor.transform(batch_tensor)
        
        return reduced_tensor
