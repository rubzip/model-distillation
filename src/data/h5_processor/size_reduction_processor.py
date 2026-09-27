from pathlib import Path
from typing import Union

import joblib
import numpy as np
import torch
from sklearn.decomposition import IncrementalPCA
from torch.utils.data import DataLoader

from .base_processor import AbstractH5BatchProcessor
from src.processing.base import BaseProcessing




class H5SizeReductionProcessor(AbstractH5BatchProcessor):
    def __init__(
        self,
        input_h5_path: Union[Path, str],
        output_h5_path: Union[Path, str],
        reducer: BaseProcessing,  # Inyectamos la dependencia
        input_dataset_name: str = "data",
        output_dataset_name: str = "data",
        batch_size: int = 1024,
        device: torch.device = torch.device("cpu"),
    ):
        # Validación temprana: Protegemos el pipeline asegurando que el modelo esté listo
        if not reducer.is_fitted:
            raise ValueError(
                "El reductor debe estar entrenado (is_fitted=True) antes de procesar el HDF5."
            )

        super().__init__(
            input_h5_path=input_h5_path,
            output_h5_path=output_h5_path,
            processor=reducer,
            input_dataset_name=input_dataset_name,
            output_dataset_name=output_dataset_name,
            batch_size=batch_size,
            device=device,
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
