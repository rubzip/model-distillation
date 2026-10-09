from pathlib import Path
from typing import Union

import joblib
import numpy as np
import torch
from sklearn.decomposition import IncrementalPCA

from ..base import BaseProcessing


class PCAEmbeddingReducer(BaseProcessing):
    def __init__(self, target_dim: int = 32):
        self.target_dim = target_dim
        self.pca = IncrementalPCA(n_components=target_dim)
        self.is_fitted = False

    def to(self, device: torch.device) -> 'PCAEmbeddingReducer':
        return self

    def fit(self, h5_path: Union[str, Path], batch_size: int = 1024, dataset_key: str = "data") -> None:
        import h5py
        print(f"Ajustando PCA (Target Dim: {self.target_dim})...")
        buffer = []
        fitted_batches = 0

        def data_iterator():
            with h5py.File(h5_path, 'r') as f:
                data = f[dataset_key]
                n_samples = data.shape[0]
                for i in range(0, n_samples, batch_size):
                    yield data[i:i + batch_size]

        iterator = data_iterator()

        for embeddings in iterator:
            if isinstance(embeddings, torch.Tensor):
                embeddings = embeddings.detach().cpu().numpy()

            if embeddings.ndim != 2:
                raise ValueError(
                    f"Se esperaba array 2D [Batch, Dim], pero se recibió forma {embeddings.shape}"
                )

            buffer.append(embeddings)
            total_buffered = sum(b.shape[0] for b in buffer)

            # IncrementalPCA requiere que cada lote tenga al menos target_dim muestras
            if total_buffered >= self.target_dim:
                merged = np.concatenate(buffer, axis=0) if len(buffer) > 1 else buffer[0]
                self.pca.partial_fit(merged)
                fitted_batches += 1
                buffer.clear()

        # Si quedaron muestras pendientes en el buffer
        if buffer:
            total_buffered = sum(b.shape[0] for b in buffer)
            if total_buffered >= self.target_dim:
                merged = np.concatenate(buffer, axis=0) if len(buffer) > 1 else buffer[0]
                self.pca.partial_fit(merged)
                fitted_batches += 1
            elif fitted_batches == 0:
                raise ValueError(
                    f"Se requieren al menos {self.target_dim} muestras totales para ajustar PCA, "
                    f"pero solo se encontraron {total_buffered}."
                )
            else:
                print(
                    f"Aviso: Se omitieron {total_buffered} muestras residuales finales (menor a target_dim={self.target_dim})."
                )

        self.is_fitted = True
        print(f"PCA Ajustado. Varianza retenida: {sum(self.get_retained_variance()):.4f}")

    def get_retained_variance(self) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("El PCA no está entrenado.")
        return self.pca.explained_variance_ratio_

    def save(self, filepath: Union[Path, str]) -> None:
        """Guarda el modelo ajustado a disco."""
        if not self.is_fitted:
            raise ValueError("No se puede guardar un PCA que no está ajustado.")
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"pca": self.pca, "target_dim": self.target_dim, "is_fitted": self.is_fitted}, filepath
        )
        print(f"PCA guardado en {filepath}")

    def load(self, filepath: Union[Path, str]) -> None:
        """Carga un modelo previamente ajustado desde disco."""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Archivo PCA no encontrado: {filepath}")
        state = joblib.load(filepath)
        self.pca = state["pca"]
        self.target_dim = state["target_dim"]
        self.is_fitted = state.get("is_fitted", True)
        print(f"PCA cargado desde {filepath}")

    @classmethod
    def from_file(cls, filepath: Union[Path, str]) -> "PCAEmbeddingReducer":
        """Instancia y carga un reducer directamente desde archivo."""
        instance = cls(target_dim=1)
        instance.load(filepath)
        return instance

    def process(self, e: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        if not self.is_fitted:
            raise ValueError("El PCA no está entrenado. Llama a fit() o load() primero.")

        is_tensor = isinstance(e, torch.Tensor)
        device = e.device if is_tensor else None

        # Convertir a numpy para sklearn
        e_np = e.detach().cpu().numpy() if is_tensor else e

        reduced_np = self.pca.transform(e_np)

        # Devolver en el mismo formato que entró
        if is_tensor:
            return torch.from_numpy(reduced_np).to(device)
        return reduced_np
