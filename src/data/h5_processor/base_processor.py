import math
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union

import h5py
import torch

from src.processing.base import BaseProcessing


class AbstractH5BatchProcessor(ABC):
    def __init__(
        self,
        input_h5_path: Union[Path, str],
        output_h5_path: Union[Path, str],
        processor: BaseProcessing,
        input_dataset_name: str = "audio_segments",
        output_dataset_name: str = "data",
        batch_size: int = 512,
        device: torch.device = torch.device("cpu"),
    ):
        self.input_h5_path = Path(input_h5_path)
        self.output_h5_path = Path(output_h5_path)

        self.processor = processor.to(device)
        self.processor.eval()

        self.input_dataset_name = input_dataset_name
        self.output_dataset_name = output_dataset_name
        self.batch_size = batch_size
        self.device = device

        self.output_h5_path.parent.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def _process_batch(self, batch_tensor: torch.Tensor) -> torch.Tensor:
        """
        Child classes MUST implement this method.
        Takes a tensor from the HDF5 and returns the processed tensor.
        """
        pass

    def _get_input_dataset(self, h5_in: h5py.File) -> h5py.Dataset:
        if self.input_dataset_name in h5_in:
            return h5_in[self.input_dataset_name]
        available = list(h5_in.keys())
        if len(available) == 1:
            print(
                f" -> Warning: '{self.input_dataset_name}' not found. Using only available dataset: '{available[0]}'"
            )
            return h5_in[available[0]]
        raise KeyError(
            f"Dataset '{self.input_dataset_name}' not found in {self.input_h5_path}. Available: {available}"
        )

    def _infer_shape(self, h5_in: h5py.File) -> tuple:
        """Generic test pass to discover output dimensions."""
        input_dataset = self._get_input_dataset(h5_in)
        dummy_np = input_dataset[0:1]  # Load 1 single example
        dummy_tensor = torch.from_numpy(dummy_np).to(self.device)

        with torch.no_grad():
            dummy_out = self._process_batch(dummy_tensor)

        return tuple(dummy_out.shape[1:])  # Ignore the batch dimension (0)

    @torch.no_grad()
    def run(self) -> None:
        """The common orchestrator that handles all massive I/O."""
        print(f"\n[{self.__class__.__name__}] Starting...")
        print(f" -> Input:  {self.input_h5_path}")
        print(f" -> Output: {self.output_h5_path}")

        with (
            h5py.File(self.input_h5_path, "r") as h5_in,
            h5py.File(self.output_h5_path, "w") as h5_out,
        ):
            input_dataset = self._get_input_dataset(h5_in)
            total_items = input_dataset.shape[0]

            # Shape inference and pre-allocation
            output_shape = self._infer_shape(h5_in)
            final_shape = (total_items,) + output_shape
            print(f" -> Detected shape: {final_shape}")

            # Configure optimized chunking for I/O and DataLoader
            chunk_batch = min(self.batch_size, 128, max(total_items, 1))
            chunks = (chunk_batch,) + output_shape

            output_dataset = h5_out.create_dataset(
                self.output_dataset_name,
                shape=final_shape,
                dtype="float32",
                compression="lzf",
                chunks=chunks,
            )
            h5_out.attrs["total_items"] = total_items

            num_batches = math.ceil(total_items / self.batch_size)

            for i in range(num_batches):
                start_idx = i * self.batch_size
                end_idx = min(start_idx + self.batch_size, total_items)

                # Read from disk to GPU/CPU
                batch_np = input_dataset[start_idx:end_idx]
                batch_tensor = torch.from_numpy(batch_np).to(self.device)

                # CHILD CLASS LOGIC IS CALLED HERE!
                processed_tensor = self._process_batch(batch_tensor)

                # From GPU/CPU to disk
                output_dataset[start_idx:end_idx] = processed_tensor.cpu().numpy()

                # Progress
                if (i + 1) % 10 == 0 or i == num_batches - 1:
                    print(
                        f"\r -> Batch {i + 1}/{num_batches} ({(end_idx / total_items) * 100:.1f}%)",
                        end="",
                    )

        print("\n -> Process finished successfully!")
