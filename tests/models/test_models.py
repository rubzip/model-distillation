import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn

from src.models import (
    DistillationModel,
    get_operations_MFlop,
    get_size_MBytes,
    load_distillation_model,
    save_distillation_model,
)
from src.models.architectures.classifier import MLPFactory
from src.models.architectures.embedding import DumbEmbedding


class TestDistillationModel(unittest.TestCase):
    def setUp(self):
        self.embed = DumbEmbedding(input_shape=(1, 16), embedding_dim=8)
        self.cls = MLPFactory.create_mlp(input_size=8, hidden_sizes=[16], num_classes=3)
        self.model = DistillationModel(self.embed, self.cls)

    def test_init_validation(self):
        with self.assertRaises(TypeError):
            DistillationModel("not_a_module", self.cls)
        with self.assertRaises(TypeError):
            DistillationModel(self.embed, "not_a_module")

    def test_forward(self):
        x = torch.randn(4, 1, 16)
        e, y = self.model(x)
        self.assertEqual(e.shape, (4, 8))
        self.assertEqual(y.shape, (4, 3))

    def test_classify(self):
        x = torch.randn(4, 1, 16)
        preds = self.model.classify(x)
        self.assertEqual(preds.shape, (4,))
        self.assertEqual(preds.dtype, torch.int64)


class TestModelIO(unittest.TestCase):
    def setUp(self):
        self.embed = DumbEmbedding(input_shape=(1, 8), embedding_dim=4)
        self.cls = MLPFactory.create_mlp(input_size=4, hidden_sizes=[], num_classes=2)
        self.model = DistillationModel(self.embed, self.cls)

    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            save_path = Path(tmp_dir) / "nested" / "model.pt"
            save_distillation_model(self.model, save_path)
            self.assertTrue(save_path.exists())

            new_model = DistillationModel(
                DumbEmbedding(input_shape=(1, 8), embedding_dim=4),
                MLPFactory.create_mlp(input_size=4, hidden_sizes=[], num_classes=2),
            )
            loaded = load_distillation_model(new_model, save_path, device="cpu")
            self.assertIs(loaded, new_model)

            # Verify identical weights
            for p1, p2 in zip(self.model.parameters(), loaded.parameters()):
                self.assertTrue(torch.equal(p1, p2))


class TestPerformance(unittest.TestCase):
    def setUp(self):
        self.layer = nn.Linear(10, 5)

    def test_get_size_mbytes(self):
        size = get_size_MBytes(self.layer)
        self.assertIsInstance(size, float)
        self.assertGreater(size, 0.0)

    def test_get_operations_mflop(self):
        mflops = get_operations_MFlop(self.layer, input_shape=(1, 10))
        self.assertIsInstance(mflops, float)
        self.assertGreater(mflops, 0.0)


class TestArchitectures(unittest.TestCase):
    def test_dumb_embedding_shapes(self):
        # 1D input shape
        embed1d = DumbEmbedding(input_shape=(16,), embedding_dim=8)
        out1d = embed1d(torch.randn(2, 16))
        self.assertEqual(out1d.shape, (2, 8))

        # 2D input shape
        embed2d = DumbEmbedding(input_shape=(1, 16), embedding_dim=8)
        out2d = embed2d(torch.randn(2, 1, 16))
        self.assertEqual(out2d.shape, (2, 8))

        # 3D input shape
        embed3d = DumbEmbedding(input_shape=(1, 4, 4), embedding_dim=8)
        out3d = embed3d(torch.randn(2, 1, 4, 4))
        self.assertEqual(out3d.shape, (2, 8))

    def test_mlp_factory(self):
        mlp = MLPFactory.create_mlp(input_size=10, hidden_sizes=[20, 15], num_classes=4)
        out = mlp(torch.randn(3, 10))
        self.assertEqual(out.shape, (3, 4))

        # Empty hidden sizes (direct linear classifier)
        linear_cls = MLPFactory.create_mlp(input_size=10, hidden_sizes=[], num_classes=4)
        self.assertEqual(linear_cls(torch.randn(3, 10)).shape, (3, 4))

    def test_mlp_factory_validation(self):
        with self.assertRaises(ValueError):
            MLPFactory.create_mlp(input_size=0, hidden_sizes=[10], num_classes=2)
        with self.assertRaises(ValueError):
            MLPFactory.create_mlp(input_size=10, hidden_sizes=[10], num_classes=0)
        with self.assertRaises(ValueError):
            MLPFactory.create_mlp(input_size=10, hidden_sizes=[-1], num_classes=2)
        with self.assertRaises(ValueError):
            MLPFactory.create_mlp(
                input_size=10, hidden_sizes=[10], num_classes=2, dropout_rate=-0.1
            )
        with self.assertRaises(ValueError):
            MLPFactory.create_mlp(input_size=10, hidden_sizes=[10], num_classes=2, dropout_rate=1.0)


if __name__ == "__main__":
    unittest.main()
