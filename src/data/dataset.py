from pathlib import Path
from typing import Callable, Dict, List, Optional
import h5py
import torch
from torch.utils.data import Dataset

from .metadata import AudioSegmentMetadata

class KnowledgeDistillationDataset(Dataset):
    def __init__(
        self, 
        metadata_list: List[AudioSegmentMetadata], 
        features_h5_path: Path | str, 
        label_map: Dict[str, int],
        embeddings_h5_path: Optional[Path | str] = None, 
        features_dataset_name: str = "data",
        embeddings_dataset_name: str = "data",
        indices: Optional[List[int]] = None,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None
    ):
        """
        Dataset que une características de audio, embeddings precomputados y metadatos.
        Diseñado para ser seguro con multiprocesamiento (num_workers > 0) y subconjuntos (train/val/test).

        Args:
            metadata_list: Lista de metadatos de los segmentos (puede ser un subconjunto filtrado).
            features_h5_path: Ruta al archivo HDF5 con las características (espectrogramas).
            label_map: Diccionario {categoría: índice_entero}.
            embeddings_h5_path: Ruta opcional al HDF5 de embeddings de profesor/reducidos.
            features_dataset_name: Nombre del dataset en features_h5.
            embeddings_dataset_name: Nombre del dataset en embeddings_h5.
            indices: Mapeo explícito opcional de índices hacia filas del HDF5.
                     Si no se proporciona, se infiere automáticamente a partir de int(meta.id).
            transform: Función de transformación/aumento de datos opcional aplicada a 'x'.
        """
        self.metadata_list = metadata_list
        self.features_h5_path = Path(features_h5_path)
        self.embeddings_h5_path = Path(embeddings_h5_path) if embeddings_h5_path is not None else None
        self.label_map = label_map
        
        self.features_dataset_name = features_dataset_name
        self.embeddings_dataset_name = embeddings_dataset_name
        self.transform = transform

        # Precalcular el mapeo de índice local -> índice de fila HDF5
        if indices is not None:
            if len(indices) != len(metadata_list):
                raise ValueError(f"Longitud de indices ({len(indices)}) no coincide con metadata_list ({len(metadata_list)})")
            self._h5_indices = list(indices)
        else:
            self._h5_indices = []
            for i, meta in enumerate(metadata_list):
                try:
                    self._h5_indices.append(int(meta.id))
                except (ValueError, TypeError):
                    self._h5_indices.append(i)

        # Apertura perezosa por proceso/worker para compatibilidad con DataLoader
        self.h5_features = None
        self.h5_embeddings = None

    def __len__(self) -> int:
        return len(self.metadata_list)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Apertura perezosa de los archivos HDF5
        if self.h5_features is None:
            self.h5_features = h5py.File(self.features_h5_path, 'r', swmr=True)
        if self.h5_embeddings is None and self.embeddings_h5_path is not None:
            self.h5_embeddings = h5py.File(self.embeddings_h5_path, 'r', swmr=True)

        h5_row = self._h5_indices[idx]

        # 1. Obtener X (Características: STFT, MelSpec, etc.)
        x_np = self.h5_features[self.features_dataset_name][h5_row]
        x = torch.from_numpy(x_np)

        if self.transform is not None:
            x = self.transform(x)

        # 2. Obtener E (Embedding precomputado o reducido por PCA)
        if self.h5_embeddings is not None:
            e_np = self.h5_embeddings[self.embeddings_dataset_name][h5_row]
            e = torch.from_numpy(e_np)
        else:
            e = torch.empty(0, dtype=torch.float32)

        # 3. Obtener Y (Etiqueta de clasificación)
        meta = self.metadata_list[idx]
        y_int = self.label_map[meta.category]
        y = torch.tensor(y_int, dtype=torch.long)

        return x, e, y

    def close(self):
        """Cierra de forma segura los archivos HDF5 abiertos."""
        if self.h5_features is not None:
            try:
                self.h5_features.close()
            except Exception:
                pass
            self.h5_features = None
        if self.h5_embeddings is not None:
            try:
                self.h5_embeddings.close()
            except Exception:
                pass
            self.h5_embeddings = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        self.close()
