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
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ):
        """
        Dataset that merges audio features, precomputed embeddings, and metadata.
        Designed to be multiprocessing-safe (num_workers > 0) and split-aware (train/val/test).

        Args:
            metadata_list: List of segment metadata (can be a filtered subset).
            features_h5_path: Path to the HDF5 file containing the features (spectrograms).
            label_map: Dictionary {category: integer_index}.
            embeddings_h5_path: Optional path to the teacher/reduced embeddings HDF5.
            features_dataset_name: Dataset name in features_h5.
            embeddings_dataset_name: Dataset name in embeddings_h5.
            indices: Optional explicit mapping of indices to HDF5 rows.
                     If not provided, inferred automatically from int(meta.id).
            transform: Optional data transformation/augmentation function applied to 'x'.
        """
        # Lazy opening per process/worker for DataLoader compatibility
        self.h5_features = None
        self.h5_embeddings = None

        self.features_h5_path = Path(features_h5_path)
        self.embeddings_h5_path = (
            Path(embeddings_h5_path) if embeddings_h5_path is not None else None
        )
        self.features_dataset_name = features_dataset_name
        self.embeddings_dataset_name = embeddings_dataset_name
        self.transform = transform
        self._length = len(metadata_list)

        # Extract labels into a lightweight tensor and discard the heavy list
        self.labels = torch.tensor(
            [label_map[meta.category] for meta in metadata_list], 
            dtype=torch.long
        )

        # Precompute local index -> HDF5 row index mapping
        if indices is not None:
            if len(indices) != len(metadata_list):
                raise ValueError(
                    f"Length of indices ({len(indices)}) does not match metadata_list ({len(metadata_list)})"
                )
            self._h5_indices = list(indices)
        else:
            self._h5_indices = []
            for i, meta in enumerate(metadata_list):
                try:
                    self._h5_indices.append(int(meta.id))
                except (ValueError, TypeError):
                    self._h5_indices.append(i)

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Lazy opening of HDF5 files
        if self.h5_features is None:
            self.h5_features = h5py.File(self.features_h5_path, "r", swmr=True)
        if self.h5_embeddings is None and self.embeddings_h5_path is not None:
            self.h5_embeddings = h5py.File(self.embeddings_h5_path, "r", swmr=True)

        h5_row = self._h5_indices[idx]

        # 1. Get X (Features: STFT, MelSpec, etc.)
        x_np = self.h5_features[self.features_dataset_name][h5_row]
        x = torch.from_numpy(x_np)

        if self.transform is not None:
            x = self.transform(x)

        # 2. Get E (Precomputed or PCA reduced embedding)
        if self.h5_embeddings is not None:
            e_np = self.h5_embeddings[self.embeddings_dataset_name][h5_row]
            e = torch.from_numpy(e_np)
        else:
            e = torch.empty(0, dtype=torch.float32)

        # 3. Get Y (Classification label)
        y = self.labels[idx]

        return x, e, y

    def close(self):
        """Safely closes the open HDF5 files."""
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
