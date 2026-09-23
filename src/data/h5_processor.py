import h5py
import numpy as np
import torch
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Any
import torchaudio

from .metadata import AudioSegmentMetadata

class AbstractH5Processor(ABC):
    def __init__(self, output_h5_path: Path | str):
        self.output_h5_path = Path(output_h5_path)
        self.output_h5_path.parent.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def load_segment(self, meta: AudioSegmentMetadata) -> Any:
        """Carga la información requerida (desde WAV o desde un H5 previo)."""
        pass

    @abstractmethod
    def __call__(self, data: Any) -> np.ndarray:
        """Transforma los datos. DEBE devolver un numpy array para guardarse en HDF5."""
        pass

    def save_to_h5(self, h5_file: h5py.File, segment_id: str, data: np.ndarray) -> None:
        """Lógica estandarizada para guardar en HDF5 con compresión."""
        if segment_id in h5_file:
            del h5_file[segment_id]
        
        h5_file.create_dataset(
            name=segment_id, 
            data=data, 
            compression="gzip", 
            compression_opts=4
        )

    def process_dataset(self, metadata_list: List[AudioSegmentMetadata]) -> None:
        """El orquestador: Abre HDF5 una vez y procesa todo el dataset."""
        print(f"[{self.__class__.__name__}] Guardando en: {self.output_h5_path}")
        
        with h5py.File(self.output_h5_path, 'a') as h5_file:
            for i, meta in enumerate(metadata_list):
                try:
                    segment_id = meta.fname.stem  # Ej: "audio_001_seg_05"
                    
                    # 1. Cargar datos específicos del segmento
                    raw_data = self.load_segment(meta)
                    
                    # 2. Aplicar la transformación (Feature, Embed, PCA, etc.)
                    processed_data = self.__call__(raw_data)
                    
                    # 3. Guardar en el HDF5
                    self.save_to_h5(h5_file, segment_id, processed_data)
                    
                    if (i + 1) % 100 == 0:
                        print(f"  -> Procesados {i + 1}/{len(metadata_list)}")
                        
                except Exception as e:
                    print(f"Error en segmento {meta.fname}: {e}")
                    
        print(f"[{self.__class__.__name__}] Proceso finalizado.\n")
