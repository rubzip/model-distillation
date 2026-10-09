import unittest

import torch.nn as nn

from src.models import DistillationModel, validate_model, validate_output, validate_shapes


class DummyEmbeddingNet(nn.Module):
    def __init__(self, in_features=32, embed_dim=16):
        super().__init__()
        self.fc = nn.Linear(in_features, embed_dim)

    def forward(self, x):
        # x is [B, in_features] or [B, Channels, in_features]
        if x.dim() > 2:
            x = x.flatten(start_dim=1)
        return self.fc(x)


class DummyClassifierNet(nn.Module):
    def __init__(self, embed_dim=16, num_classes=3, apply_softmax=False):
        super().__init__()
        self.fc = nn.Linear(embed_dim, num_classes)
        self.apply_softmax = apply_softmax
        if apply_softmax:
            self.softmax = nn.Softmax(dim=-1)

    def forward(self, e):
        y = self.fc(e)
        if self.apply_softmax:
            y = self.softmax(y)
        return y


class TestModelValidation(unittest.TestCase):
    def setUp(self):
        self.input_shape = (32,)
        self.embed_dim = 16
        self.num_classes = 3

        self.embed_net = DummyEmbeddingNet(in_features=32, embed_dim=self.embed_dim)
        self.cls_net = DummyClassifierNet(embed_dim=self.embed_dim, num_classes=self.num_classes)
        self.model = DistillationModel(
            embedding_model=self.embed_net, classifier_model=self.cls_net
        )

    def test_validate_shapes_success(self):
        self.assertTrue(
            validate_shapes(self.model, input_shape=self.input_shape, embedding_dim=self.embed_dim)
        )

    def test_validate_shapes_dimension_mismatch(self):
        with self.assertRaises(ValueError) as ctx:
            validate_shapes(self.model, input_shape=self.input_shape, embedding_dim=64)
        self.assertIn("Embedding dimension mismatch", str(ctx.exception))

    def test_validate_output_success(self):
        self.assertTrue(
            validate_output(self.model, input_shape=self.input_shape, output_dim=self.num_classes)
        )

    def test_validate_output_class_dimension_mismatch(self):
        with self.assertRaises(ValueError) as ctx:
            validate_output(self.model, input_shape=self.input_shape, output_dim=10)
        self.assertIn("Output class dimension mismatch", str(ctx.exception))

    def test_validate_output_detects_accidental_softmax(self):
        softmax_cls = DummyClassifierNet(
            embed_dim=self.embed_dim, num_classes=self.num_classes, apply_softmax=True
        )
        bad_model = DistillationModel(embedding_model=self.embed_net, classifier_model=softmax_cls)

        with self.assertRaises(ValueError) as ctx:
            validate_output(
                bad_model,
                input_shape=self.input_shape,
                output_dim=self.num_classes,
                expected_logits=True,
            )
        self.assertIn("softmax probabilities", str(ctx.exception))

    def test_validate_model_full(self):
        self.assertTrue(
            validate_model(
                self.model,
                input_shape=self.input_shape,
                embedding_dim=self.embed_dim,
                output_dim=self.num_classes,
            )
        )

    def test_validate_model_detects_accidental_softmax_layer(self):
        softmax_cls = nn.Sequential(
            nn.Linear(self.embed_dim, self.num_classes),
            nn.Softmax(dim=-1),
        )
        bad_model = DistillationModel(embedding_model=self.embed_net, classifier_model=softmax_cls)
        with self.assertRaises(ValueError) as ctx:
            validate_model(
                bad_model,
                input_shape=self.input_shape,
                embedding_dim=self.embed_dim,
                output_dim=self.num_classes,
                expected_logits=True,
            )
        self.assertIn("last layer must NOT be Softmax or LogSoftmax", str(ctx.exception))

    def test_validate_model_detects_accidental_logsoftmax_layer(self):
        logsoftmax_cls = nn.Sequential(
            nn.Linear(self.embed_dim, self.num_classes),
            nn.LogSoftmax(dim=-1),
        )
        bad_model = DistillationModel(
            embedding_model=self.embed_net, classifier_model=logsoftmax_cls
        )
        with self.assertRaises(ValueError) as ctx:
            validate_model(
                bad_model,
                input_shape=self.input_shape,
                embedding_dim=self.embed_dim,
                output_dim=self.num_classes,
                expected_logits=True,
            )
        self.assertIn("last layer must NOT be Softmax or LogSoftmax", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
