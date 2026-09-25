import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn

from src.data.h5_processor import (
    H5EmbeddingProcessor,
    H5SegmentExtractor,
    PCAEmbeddingReducer,
)
from src.data.metadata import AudioMetadata


class MockModel3D(nn.Module):
    """Model that outputs 3D tensor [Batch, 1, Embed_Dim] to test dimension preservation."""

    def __init__(self, embed_dim=16):
        super().__init__()
        self.embed_dim = embed_dim
        self.fc = nn.Linear(100, embed_dim)

    def forward(self, x):
        # x is [B, 1, T] or [B, T]
        if x.dim() == 3:
            x = x.squeeze(1)
        sub = x[:, :100]
        out = self.fc(sub)
        return out.unsqueeze(1)  # Returns [B, 1, embed_dim]


class TestH5Processors(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.base_path = Path(self.tmp_dir.name)

        # Create dummy wav
        self.sr = 16000
        self.audio_path = self.base_path / "test.wav"
        t = np.linspace(0, 2.0, self.sr * 2, False)
        sine = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
        sf.write(self.audio_path, sine, self.sr)

        self.meta = AudioMetadata(
            fname=self.audio_path,
            category="park",
            subcategory="outdoor",
            steps=len(sine),
            sr=self.sr,
            channels=1,
            location_id="loc_1",
            device_id="dev_a",
            split="TRAIN",
        )

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_segment_extractor(self):
        h5_out = self.base_path / "segments.h5"
        meta_out = self.base_path / "meta.json"

        extractor = H5SegmentExtractor(
            output_h5_path=h5_out,
            output_meta_path=meta_out,
            target_sr=self.sr,
            segment_steps=16000,
            write_batch_size=2,
        )
        metas = extractor.run([self.meta])
        self.assertEqual(len(metas), 2)
        self.assertTrue(h5_out.exists())
        self.assertTrue(meta_out.exists())

        with h5py.File(h5_out, "r") as f:
            self.assertIn("audio_segments", f)
            self.assertEqual(f["audio_segments"].shape, (2, 16000))
            self.assertEqual(f.attrs["target_sr"], 16000)

    def test_embedding_processor_preserves_dimensions_with_singleton_batch(self):
        # 1. Create a dummy segments file with 3 items
        seg_h5 = self.base_path / "test_seg.h5"
        with h5py.File(seg_h5, "w") as f:
            f.create_dataset("audio_segments", data=np.random.randn(3, 1000).astype(np.float32))

        out_h5 = self.base_path / "embeddings.h5"
        model = MockModel3D(embed_dim=16)

        processor = H5EmbeddingProcessor(
            input_h5_path=seg_h5,
            output_h5_path=out_h5,
            processor=model,
            batch_size=1,  # Batch size 1 tests that batch dim is not collapsed
        )
        processor.run()

        with h5py.File(out_h5, "r") as f:
            self.assertIn("data", f)
            # Shape should be (3, 16), NOT (3,) or 1D
            self.assertEqual(f["data"].shape, (3, 16))

    def test_pca_reducer_fit_buffering_and_persistence(self):
        reducer = PCAEmbeddingReducer(target_dim=4)

        # Batches where each batch is smaller than target_dim (e.g. 2 samples per batch)
        small_batches = [
            torch.randn(2, 8),
            torch.randn(2, 8),
            torch.randn(3, 8),
            torch.randn(1, 8),  # Total 8 samples >= target_dim=4
        ]
        # Should not raise ValueError even though individual batches < target_dim
        reducer.fit(small_batches)
        self.assertTrue(reducer.is_fitted)

        # Test transform
        sample = torch.randn(5, 8)
        transformed = reducer.transform(sample)
        self.assertIsInstance(transformed, torch.Tensor)
        self.assertEqual(transformed.shape, (5, 4))

        # Test save and load
        save_path = self.base_path / "pca_model.joblib"
        reducer.save(save_path)
        self.assertTrue(save_path.exists())

        loaded_reducer = PCAEmbeddingReducer.from_file(save_path)
        self.assertTrue(loaded_reducer.is_fitted)
        self.assertEqual(loaded_reducer.target_dim, 4)

        loaded_transformed = loaded_reducer.transform(sample)
        self.assertTrue(torch.allclose(transformed, loaded_transformed))


if __name__ == "__main__":
    unittest.main()
