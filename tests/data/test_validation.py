import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from src.data.dataset import KnowledgeDistillationDataset
from src.data.metadata import AudioSegmentMetadata
from src.data.validation import DatasetValidator, check_split_leakage, validate_dataset


class TestValidation(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.base_path = Path(self.tmp_dir.name)

        self.features_h5 = self.base_path / "feat.h5"
        self.embeddings_h5 = self.base_path / "embed.h5"

        # Create valid mock HDF5 datasets with 5 samples
        feats = np.random.randn(5, 64, 32).astype(np.float32)
        embeds = np.random.randn(5, 16).astype(np.float32)

        with h5py.File(self.features_h5, "w") as f:
            f.create_dataset("data", data=feats)
        with h5py.File(self.embeddings_h5, "w") as f:
            f.create_dataset("data", data=embeds)

        self.label_map = {"park": 0, "street": 1}
        self.valid_metas = [
            AudioSegmentMetadata(
                id=f"000000{i}",
                original_fname=f"audio_{i}.wav",
                start_step=0,
                end_step=16000,
                sr=16000,
                split="TRAIN" if i < 3 else "VAL",
                category="park" if i % 2 == 0 else "street",
                location_id=f"loc_{i}",
            )
            for i in range(5)
        ]

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_valid_dataset_passes(self):
        dataset = KnowledgeDistillationDataset(
            metadata_list=self.valid_metas,
            features_h5_path=self.features_h5,
            embeddings_h5_path=self.embeddings_h5,
            label_map=self.label_map,
        )
        validator = DatasetValidator(dataset)
        self.assertTrue(validator.validate(num_samples=5))
        self.assertTrue(validate_dataset(dataset, num_samples=5))
        dataset.close()

    def test_detects_unmapped_category(self):
        invalid_metas = list(self.valid_metas)
        invalid_metas[0] = AudioSegmentMetadata(
            id="0000000",
            original_fname="audio_0.wav",
            start_step=0,
            end_step=16000,
            sr=16000,
            split="TRAIN",
            category="beach",  # Not in label_map
        )
        with self.assertRaises(KeyError):
            _ = KnowledgeDistillationDataset(
                metadata_list=invalid_metas,
                features_h5_path=self.features_h5,
                embeddings_h5_path=self.embeddings_h5,
                label_map=self.label_map,
            )

    def test_detects_out_of_bounds_index(self):
        invalid_metas = list(self.valid_metas)
        invalid_metas[0] = AudioSegmentMetadata(
            id="0000099",
            original_fname="audio_0.wav",
            start_step=0,
            end_step=16000,
            sr=16000,
            split="TRAIN",
            category="park",
        )
        dataset = KnowledgeDistillationDataset(
            metadata_list=invalid_metas,
            features_h5_path=self.features_h5,
            embeddings_h5_path=self.embeddings_h5,
            label_map=self.label_map,
        )
        validator = DatasetValidator(dataset)
        with self.assertRaises(IndexError):
            validator.check_indices()
        dataset.close()

    def test_detects_nan_values(self):
        nan_feats_h5 = self.base_path / "nan_feats.h5"
        corrupt = np.random.randn(5, 64, 32).astype(np.float32)
        corrupt[1, 0, 0] = np.nan
        with h5py.File(nan_feats_h5, "w") as f:
            f.create_dataset("data", data=corrupt)

        dataset = KnowledgeDistillationDataset(
            metadata_list=self.valid_metas, features_h5_path=nan_feats_h5, label_map=self.label_map
        )
        validator = DatasetValidator(dataset)
        with self.assertRaises(ValueError):
            validator.check_samples(num_samples=5)
        dataset.close()

    def test_split_leakage_detection(self):
        train_metas = [
            AudioSegmentMetadata(
                id="0000000",
                original_fname="audio_shared.wav",
                start_step=0,
                end_step=16000,
                sr=16000,
                split="TRAIN",
                category="park",
                location_id="loc_1",
            )
        ]
        val_metas = [
            AudioSegmentMetadata(
                id="0000001",
                original_fname="audio_shared.wav",  # Leak!
                start_step=16000,
                end_step=32000,
                sr=16000,
                split="VAL",
                category="park",
                location_id="loc_1",
            )
        ]

        with self.assertRaises(ValueError) as ctx:
            check_split_leakage(train_metas, val_metas)
        self.assertIn("Data leakage", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
