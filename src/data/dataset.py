import h5py
import torch
from torch.utils.data import Dataset
from typing import List, Dict
from pathlib import Path

# Importas tu dataclass
from .metadata import AudioSegmentMetadata

class KnowledgeDistillationDataset(Dataset):
    def __init__(
        self, 
        metadata_list: List[AudioSegmentMetadata], 
        features_h5_path: Path | str, 
        embeddings_h5_path: Path | str, 
        label_map: Dict[str, int],
        features_dataset_name: str = "data",
        embeddings_dataset_name: str = "data"
    ):
        """
        Dataset que une características de audio, embeddings precomputados y metadatos.
        Diseñado para ser seguro con multiprocesamiento (num_workers > 0).
        """
        self.metadata_list = metadata_list
        self.features_h5_path = Path(features_h5_path)
        self.embeddings_h5_path = Path(embeddings_h5_path)
        self.label_map = label_map
        
        self.features_dataset_name = features_dataset_name
        self.embeddings_dataset_name = embeddings_dataset_name
        
        # Inicializamos los punteros a None para abrirlos perezosamente (Lazy Loading).
        # ESTO ES CRÍTICO: Si los abres aquí en el __init__, y usas DataLoader con 
        # num_workers > 0, PyTorch clonará el proceso y HDF5 lanzará un error de bloqueo.
        self.h5_features = None
        self.h5_embeddings = None

    def __len__(self) -> int:
        return len(self.metadata_list)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Apertura perezosa de los archivos HDF5 (se ejecuta solo en la primera llamada)
        if self.h5_features is None:
            self.h5_features = h5py.File(self.features_h5_path, 'r')
            self.h5_embeddings = h5py.File(self.embeddings_h5_path, 'r')

        # 1. Obtener X (Características: STFT, MelSpec, etc.)
        x_np = self.h5_features[self.features_dataset_name][idx]
        x = torch.from_numpy(x_np)

        # 2. Obtener E (Embedding precomputado o reducido por PCA)
        e_np = self.h5_embeddings[self.embeddings_dataset_name][idx]
        e = torch.from_numpy(e_np)

        # 3. Obtener Y (Etiqueta de clasificación)
        meta = self.metadata_list[idx]
        y_int = self.label_map[meta.category]
        y = torch.tensor(y_int, dtype=torch.long)

        return x, e, y

    def close(self):
        """Método opcional para cerrar manualmente los archivos si fuera necesario."""
        if self.h5_features is not None:
            self.h5_features.close()
        if self.h5_embeddings is not None:
            self.h5_embeddings.close()
