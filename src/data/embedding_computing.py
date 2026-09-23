import h5py
import numpy as np
import torch
import torchaudio
from pathlib import Path
from typing import Callable, List, Union

# Asumimos que AudioSegmentMetadata viene de tu archivo de metadata
from .metadata import AudioSegmentMetadata

class EmbeddingProcessor:
    def __init__(
        self, 
        processor: Callable, 
        output_h5_path: Union[str, Path], 
        device: torch.device = torch.device("cpu")
    ):
        """
        Toma audios segmentados, extrae sus embeddings usando un modelo (processor) 
        y los almacena de forma masiva y comprimida en un único archivo HDF5.
        
        Args:
            processor (Callable): Un modelo de PyTorch o función que toma un waveform y devuelve un embedding.
            output_h5_path (str | Path): Ruta donde se guardará el archivo .h5 (ej. 'data/embeddings.h5')
            device (torch.device): Dispositivo para cargar el modelo y procesar (CPU/CUDA).
        """
        self.processor = processor
        self.output_h5_path = Path(output_h5_path)
        self.device = device
        
        # Nos aseguramos de que la carpeta de destino exista
        self.output_h5_path.parent.mkdir(parents=True, exist_ok=True)

    def _load_audio(self, filepath: Path) -> torch.Tensor:
        """Carga el audio desde el disco y lo pasa al dispositivo adecuado."""
        # torchaudio soporta pathlib.Path de forma nativa
        waveform, _ = torchaudio.load(filepath)
        return waveform.to(self.device)

    @torch.no_grad()
    def _extract_embedding(self, waveform: torch.Tensor) -> np.ndarray:
        """Genera el embedding y lo convierte a un array de NumPy para HDF5."""
        embedding = self.processor(waveform)
        
        # HDF5 requiere arrays de numpy, no tensores de PyTorch
        if isinstance(embedding, torch.Tensor):
            # Si el tensor está en GPU o requiere gradientes, lo preparamos
            embedding = embedding.detach().cpu().numpy()
            
        return embedding

    def _save_to_hdf5(self, h5_file: h5py.File, dataset_key: str, embedding: np.ndarray) -> None:
        """Guarda el array en el archivo HDF5 usando compresión para ahorrar espacio."""
        # Si el segmento ya existe (por si re-ejecutas el código), lo borramos para sobreescribir
        if dataset_key in h5_file:
            del h5_file[dataset_key]
            
        # Creamos un dataset. HDF5 permite compresión gzip, que es genial para embeddings
        h5_file.create_dataset(
            name=dataset_key, 
            data=embedding, 
            compression="gzip", 
            compression_opts=4  # 4 es un buen balance entre velocidad y compresión
        )

    def process_dataset(self, metadata_list: List[AudioSegmentMetadata]) -> None:
        """
        Abre el archivo HDF5 una sola vez (para máxima velocidad) y procesa 
        toda la lista de metadatos guardando el resultado.
        """
        print(f"Abriendo archivo HDF5 para escritura en: {self.output_h5_path}")
        
        # 'a' (append) crea el archivo si no existe, o añade datos si ya existe
        with h5py.File(self.output_h5_path, 'a') as h5_file:
            
            for i, meta in enumerate(metadata_list):
                try:
                    # 1. Usamos el nombre del archivo (sin extensión) como llave única (ej: 'audio_123_seg_001')
                    segment_id = meta.fname.stem
                    
                    # 2. Cargar audio
                    waveform = self._load_audio(meta.fname)
                    
                    # 3. Extraer características
                    embedding = self._extract_embedding(waveform)
                    
                    # 4. Guardar en HDF5
                    self._save_to_hdf5(h5_file, segment_id, embedding)
                    
                    if (i + 1) % 100 == 0:
                        print(f"Procesados {i + 1}/{len(metadata_list)} segmentos...")
                        
                except Exception as e:
                    print(f"Error procesando el archivo {meta.fname}: {e}")
                    
        print("¡Procesamiento masivo completado con éxito!")
