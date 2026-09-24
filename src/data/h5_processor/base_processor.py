import math
import h5py
import torch
import numpy as np
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Union

class AbstractH5BatchProcessor(ABC):
    def __init__(
        self,
        input_h5_path: Union[Path, str],
        output_h5_path: Union[Path, str],
        processor: Union[Callable, torch.nn.Module],
        input_dataset_name: str = "audio_segments",
        output_dataset_name: str = "data",
        batch_size: int = 512,
        device: torch.device = torch.device("cpu")
    ):
        self.input_h5_path = Path(input_h5_path)
        self.output_h5_path = Path(output_h5_path)
        
        # Si es un modelo de PyTorch, lo pasamos al device y lo ponemos en eval()
        if isinstance(processor, torch.nn.Module):
            self.processor = processor.to(device).eval()
        else:
            self.processor = processor
            
        self.input_dataset_name = input_dataset_name
        self.output_dataset_name = output_dataset_name
        self.batch_size = batch_size
        self.device = device

        self.output_h5_path.parent.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def _process_batch(self, batch_tensor: torch.Tensor) -> torch.Tensor:
        """
        Las clases hijas DEBEN implementar este método.
        Toma un tensor del HDF5 y devuelve el tensor procesado.
        """
        pass

    def _infer_shape(self, h5_in: h5py.File) -> tuple:
        """Pase de prueba genérico para descubrir las dimensiones de salida."""
        dummy_np = h5_in[self.input_dataset_name][0:1] # Cargar 1 solo ejemplo
        dummy_tensor = torch.from_numpy(dummy_np).to(self.device)
        
        with torch.no_grad():
            dummy_out = self._process_batch(dummy_tensor)
            
        return tuple(dummy_out.shape[1:]) # Ignorar la dimensión de batch (la 0)

    @torch.no_grad()
    def run(self) -> None:
        """El orquestador común que maneja todo el I/O masivo."""
        print(f"\n[{self.__class__.__name__}] Iniciando...")
        print(f" -> Input:  {self.input_h5_path}")
        print(f" -> Output: {self.output_h5_path}")
        
        with h5py.File(self.input_h5_path, 'r') as h5_in, \
             h5py.File(self.output_h5_path, 'w') as h5_out:

            input_dataset = h5_in[self.input_dataset_name]
            total_items = input_dataset.shape[0]

            # Inferencia de forma y pre-asignación
            output_shape = self._infer_shape(h5_in)
            final_shape = (total_items,) + output_shape
            print(f" -> Forma detectada: {final_shape}")

            output_dataset = h5_out.create_dataset(
                self.output_dataset_name,
                shape=final_shape,
                dtype='float32',
                compression="lzf"
            )

            num_batches = math.ceil(total_items / self.batch_size)

            for i in range(num_batches):
                start_idx = i * self.batch_size
                end_idx = min(start_idx + self.batch_size, total_items)

                # Leer de disco a GPU/CPU
                batch_np = input_dataset[start_idx:end_idx]
                batch_tensor = torch.from_numpy(batch_np).to(self.device)

                # ¡AQUÍ SE LLAMA A LA LÓGICA DE LA CLASE HIJA!
                processed_tensor = self._process_batch(batch_tensor)

                # De GPU/CPU a disco
                output_dataset[start_idx:end_idx] = processed_tensor.cpu().numpy()

                # Progreso
                if (i + 1) % 10 == 0 or i == num_batches - 1:
                    print(f"\r -> Batch {i+1}/{num_batches} ({(end_idx/total_items)*100:.1f}%)", end="")
                    
        print("\n -> ¡Proceso finalizado con éxito!")
