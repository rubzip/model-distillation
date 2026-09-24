from pathlib import Path
from typing import Optional, Set
import h5py
import torch
from torch.utils.data import DataLoader

from ..dataset import KnowledgeDistillationDataset


class DatasetValidator:
    """Validator for KnowledgeDistillationDataset integrity and quality."""

    def __init__(self, dataset: KnowledgeDistillationDataset):
        self.dataset = dataset

    def check_files(self) -> None:
        """Check dataset length and HDF5 files existence on disk."""
        if len(self.dataset) == 0:
            raise ValueError("Dataset is empty (len == 0).")

        features_path = Path(self.dataset.features_h5_path)
        if not features_path.is_file():
            raise FileNotFoundError(f"Features HDF5 file not found: {features_path}")

        if self.dataset.embeddings_h5_path is not None:
            emb_path = Path(self.dataset.embeddings_h5_path)
            if not emb_path.is_file():
                raise FileNotFoundError(f"Embeddings HDF5 file not found: {emb_path}")

    def check_h5_datasets(self) -> None:
        """Verify that expected dataset keys exist inside the HDF5 files."""
        with h5py.File(self.dataset.features_h5_path, "r") as f:
            if self.dataset.features_dataset_name not in f:
                raise KeyError(
                    f"Dataset key '{self.dataset.features_dataset_name}' not found in {self.dataset.features_h5_path}. "
                    f"Available keys: {list(f.keys())}"
                )

        if self.dataset.embeddings_h5_path is not None:
            with h5py.File(self.dataset.embeddings_h5_path, "r") as f:
                if self.dataset.embeddings_dataset_name not in f:
                    raise KeyError(
                        f"Dataset key '{self.dataset.embeddings_dataset_name}' not found in {self.dataset.embeddings_h5_path}. "
                        f"Available keys: {list(f.keys())}"
                    )

    def check_indices(self) -> None:
        """Verify that all mapped segment indices are within HDF5 row boundaries."""
        with h5py.File(self.dataset.features_h5_path, "r") as f:
            total_feat_rows = f[self.dataset.features_dataset_name].shape[0]

        for i, h5_idx in enumerate(self.dataset._h5_indices):
            if h5_idx < 0 or h5_idx >= total_feat_rows:
                raise IndexError(
                    f"Index out of bounds at position {i}: mapped HDF5 row {h5_idx} "
                    f"exceeds total feature rows ({total_feat_rows})."
                )

        if self.dataset.embeddings_h5_path is not None:
            with h5py.File(self.dataset.embeddings_h5_path, "r") as f:
                total_emb_rows = f[self.dataset.embeddings_dataset_name].shape[0]

            for i, h5_idx in enumerate(self.dataset._h5_indices):
                if h5_idx < 0 or h5_idx >= total_emb_rows:
                    raise IndexError(
                        f"Index out of bounds at position {i}: mapped HDF5 row {h5_idx} "
                        f"exceeds total embedding rows ({total_emb_rows})."
                    )

    def check_labels(self) -> None:
        """Verify that label map is valid and all categories in metadata exist in it."""
        if not self.dataset.label_map:
            raise ValueError("label_map is empty.")

        unmapped = {
            m.category for m in self.dataset.metadata_list
            if m.category not in self.dataset.label_map
        }
        if unmapped:
            raise KeyError(f"Categories found in metadata but missing from label_map: {unmapped}")

    def check_samples(self, num_samples: Optional[int] = 50) -> None:
        """Probe samples for numerical validity (no NaNs, Infs, or all-zero features)."""
        total = len(self.dataset)
        count = total if num_samples is None or num_samples >= total else num_samples
        step = max(1, total // count)
        probe_indices = list(range(0, total, step))[:count]

        shapes_x: Set[tuple] = set()
        all_zeros = True

        for idx in probe_indices:
            x, e, y = self.dataset[idx]

            if not isinstance(x, torch.Tensor) or not isinstance(y, torch.Tensor):
                raise TypeError(f"Sample at index {idx} returned non-tensor output.")

            if torch.isnan(x).any() or torch.isinf(x).any():
                raise ValueError(f"NaN or Inf detected in features 'x' at index {idx}.")

            if self.dataset.embeddings_h5_path is not None and e.numel() > 0:
                if torch.isnan(e).any() or torch.isinf(e).any():
                    raise ValueError(f"NaN or Inf detected in embeddings 'e' at index {idx}.")

            if not (x == 0).all():
                all_zeros = False

            shapes_x.add(tuple(x.shape))

        if len(shapes_x) > 1:
            raise ValueError(f"Inconsistent feature shapes detected across samples: {shapes_x}")

        if all_zeros:
            raise ValueError("All inspected feature tensors contain only zeros (silent/dead features).")

    def check_dataloader(self, batch_size: int = 4) -> None:
        """Verify that PyTorch DataLoader can batch and collate samples without error."""
        loader = DataLoader(
            self.dataset,
            batch_size=min(batch_size, len(self.dataset)),
            shuffle=False,
            num_workers=0
        )
        try:
            next(iter(loader))
        except Exception as e:
            raise RuntimeError(f"DataLoader batch iteration failed: {e}") from e

    def validate(self, num_samples: Optional[int] = 50) -> bool:
        """Run all dataset integrity checks sequentially."""
        self.check_files()
        self.check_h5_datasets()
        self.check_indices()
        self.check_labels()
        self.check_samples(num_samples=num_samples)
        self.check_dataloader()
        return True


def validate_dataset(
    dataset: KnowledgeDistillationDataset,
    num_samples: Optional[int] = 50
) -> bool:
    """Convenience function to run all checks on a dataset."""
    validator = DatasetValidator(dataset)
    return validator.validate(num_samples=num_samples)


def check_split_leakage(
    train_dataset: KnowledgeDistillationDataset,
    val_dataset: KnowledgeDistillationDataset,
    test_dataset: Optional[KnowledgeDistillationDataset] = None,
) -> None:
    """Check for data leakage (shared original filenames or segment IDs) between splits."""
    train_files = {str(m.original_fname) for m in train_dataset.metadata_list if m.original_fname}
    val_files = {str(m.original_fname) for m in val_dataset.metadata_list if m.original_fname}

    train_ids = {m.id for m in train_dataset.metadata_list}
    val_ids = {m.id for m in val_dataset.metadata_list}

    file_overlap = train_files.intersection(val_files)
    if file_overlap:
        raise ValueError(f"Data leakage: train and val share original files: {file_overlap}")

    id_overlap = train_ids.intersection(val_ids)
    if id_overlap:
        raise ValueError(f"Data leakage: train and val share segment IDs: {id_overlap}")

    if test_dataset is not None:
        test_files = {str(m.original_fname) for m in test_dataset.metadata_list if m.original_fname}
        test_ids = {m.id for m in test_dataset.metadata_list}

        test_file_overlap = train_files.intersection(test_files)
        if test_file_overlap:
            raise ValueError(f"Data leakage: train and test share original files: {test_file_overlap}")

        test_id_overlap = train_ids.intersection(test_ids)
        if test_id_overlap:
            raise ValueError(f"Data leakage: train and test share segment IDs: {test_id_overlap}")
