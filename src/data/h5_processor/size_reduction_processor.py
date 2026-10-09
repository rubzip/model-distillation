from pathlib import Path
from typing import Union

import torch

from src.processing.base import BaseProcessing

from .base_processor import AbstractH5BatchProcessor


class H5SizeReductionProcessor(AbstractH5BatchProcessor):
    def __init__(
        self,
        input_h5_path: Union[Path, str],
        output_h5_path: Union[Path, str],
        reducer: BaseProcessing,  # Inject the dependency
        input_dataset_name: str = "data",
        output_dataset_name: str = "data",
        batch_size: int = 1024,
        device: torch.device = torch.device("cpu"),
    ):
        # Early validation: Protect the pipeline by ensuring the model is ready
        if not reducer.is_fitted:
            raise ValueError(
                "The reducer must be fitted (is_fitted=True) before processing the HDF5."
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
        # Data integrity validation
        if batch_tensor.dim() != 2:
            raise ValueError(
                f"The PCA reducer expects 2D tensors [Batch, Embedding_Dim], "
                f"but received a {batch_tensor.dim()}D tensor."
            )

        # Since our reducer now natively handles tensors in its process() method,
        # delegation is direct and clean.
        reduced_tensor = self.processor.process(batch_tensor)

        return reduced_tensor
