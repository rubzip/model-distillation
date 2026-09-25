import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from src.data.dataset import KnowledgeDistillationDataset
from src.data.metadata import AudioSegmentMetadata


class TestDataset(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.base_path = Path(self.tmp_dir.name)

        self.features_h5 = self.base_path / "feat.h5"
        self.embeddings_h5 = self.base_path / "embed.h5"

        # Create 10 distinct samples
        # Each row i has distinct feature values (e.g. all filled with i)
        self.total_items = 10
        feats = np.zeros((self.total_items, 64, 32), dtype=np.float32)
        embeds = np.zeros((self.total_items, 16), dtype=np.float32)
        for i in range(self.total_items):
            feats[i].fill(float(i))
            embeds[i].fill(float(i * 10))

        with h5py.File(self.features_h5, "w") as f:
            f.create_dataset("data", data=feats)
        with h5py.File(self.embeddings_h5, "w") as f:
            f.create_dataset("data", data=embeds)

        self.label_map = {"park": 0, "street": 1}

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_subset_indexing_correctness(self):
        """CRITICAL: Ensure filtered subsets (e.g. VAL) look up their TRUE H5 rows, not rows 0, 1, 2."""
        # Suppose validation set has items with row IDs 3, 7, 9
        val_metas = [
            AudioSegmentMetadata(
                id="0000003",
                original_fname="audio1.wav",
                start_step=0,
                end_step=16000,
                sr=16000,
                split="VAL",
                category="street",
            ),
            AudioSegmentMetadata(
                id="0000007",
                original_fname="audio2.wav",
                start_step=0,
                end_step=16000,
                sr=16000,
                split="VAL",
                category="park",
            ),
            AudioSegmentMetadata(
                id="0000009",
                original_fname="audio3.wav",
                start_step=0,
                end_step=16000,
                sr=16000,
                split="VAL",
                category="street",
            ),
        ]

        dataset = KnowledgeDistillationDataset(
            metadata_list=val_metas,
            features_h5_path=self.features_h5,
            embeddings_h5_path=self.embeddings_h5,
            label_map=self.label_map,
        )

        self.assertEqual(len(dataset), 3)

        # Item 0 in subset corresponds to row 3
        x0, e0, y0 = dataset[0]
        self.assertEqual(x0[0, 0].item(), 3.0)  # NOT 0.0!
        self.assertEqual(e0[0].item(), 30.0)  # NOT 0.0!
        self.assertEqual(y0.item(), 1)  # "street" -> 1

        # Item 1 in subset corresponds to row 7
        x1, e1, y1 = dataset[1]
        self.assertEqual(x1[0, 0].item(), 7.0)  # NOT 1.0!
        self.assertEqual(e1[0].item(), 70.0)  # NOT 10.0!
        self.assertEqual(y1.item(), 0)  # "park" -> 0

        # Item 2 in subset corresponds to row 9
        x2, e2, y2 = dataset[2]
        self.assertEqual(x2[0, 0].item(), 9.0)  # NOT 2.0!
        self.assertEqual(e2[0].item(), 90.0)  # NOT 20.0!
        self.assertEqual(y2.item(), 1)  # "street" -> 1

        dataset.close()

    def test_transform_application(self):
        val_metas = [
            AudioSegmentMetadata(
                id="0000001",
                original_fname="audio1.wav",
                start_step=0,
                end_step=16000,
                sr=16000,
                split="TRAIN",
                category="park",
            ),
        ]

        def add_transform(tensor):
            return tensor + 100.0

        dataset = KnowledgeDistillationDataset(
            metadata_list=val_metas,
            features_h5_path=self.features_h5,
            label_map=self.label_map,
            transform=add_transform,
        )

        x, e, y = dataset[0]
        self.assertEqual(x[0, 0].item(), 101.0)
        self.assertEqual(e.numel(), 0)  # No embeddings provided -> empty tensor
        dataset.close()


if __name__ == "__main__":
    unittest.main()
