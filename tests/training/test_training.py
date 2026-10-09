import unittest

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.models import DistillationModel
from src.training import (
    ClassificationMetrics,
    DistillationCriterion,
    EarlyStopping,
    History,
    evaluate_distillation_model,
    train_distillation_model,
)


class DummyEmbeddingNet(nn.Module):
    def __init__(self, in_features=10, embed_dim=8):
        super().__init__()
        self.fc = nn.Linear(in_features, embed_dim)

    def forward(self, x):
        return self.fc(x)


class DummyClassifierNet(nn.Module):
    def __init__(self, embed_dim=8, num_classes=3):
        super().__init__()
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, e):
        return self.fc(e)


class TestDistillationCriterion(unittest.TestCase):
    def setUp(self):
        self.criterion = DistillationCriterion(lambda_e=0.5, lambda_y=1.0)
        self.e_true = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        self.e_pred = torch.tensor([[1.0, 1.0], [3.0, 5.0]])
        self.y_pred = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
        self.y_true = torch.tensor([0, 1])

    def test_distillation_criterion_scalar(self):
        loss = self.criterion(
            e_true=self.e_true,
            e_pred=self.e_pred,
            y_true=self.y_true,
            y_pred=self.y_pred,
        )
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.dim(), 0)


class TestMetricsAndHistory(unittest.TestCase):
    def test_classification_metrics(self):
        metrics = ClassificationMetrics(num_classes=2)
        logits = torch.tensor([[2.0, 0.0], [0.0, 2.0], [3.0, -1.0], [1.0, 4.0]])
        targets = torch.tensor([0, 1, 1, 1])
        metrics.update(logits, targets)
        res = metrics.compute()

        self.assertAlmostEqual(res.accuracy, 0.75)
        cm = metrics.get_cm(normalized=True)
        self.assertEqual(cm.shape, (2, 2))

    def test_history(self):
        history = History()
        metrics = ClassificationMetrics(num_classes=2).compute()
        history.add(train_loss=0.8, val_loss=0.6, result=metrics)

        self.assertEqual(history.n_epochs, 1)
        self.assertEqual(history.train_losses, [0.8])
        self.assertEqual(history.val_losses, [0.6])

        d = history.to_dict()
        self.assertIn("train_loss", d)
        self.assertIn("val_loss", d)
        self.assertIn("accuracy", d)


class TestEarlyStopping(unittest.TestCase):
    def test_lower_is_better_min_delta_logic(self):
        """Verifies that an increase in loss is NOT recognized as improvement."""
        model = nn.Linear(2, 2)
        ea = EarlyStopping(patience=2, min_delta=0.05, higher_is_better=False, verbose=False)

        # Epoch 1: initial score 1.00
        ea(value=1.00, model=model)
        self.assertEqual(ea.best_value, 1.00)
        self.assertEqual(ea.epochs_without_improvement, 0)

        # Epoch 2: loss slightly increases to 1.02 -> should NOT be improvement!
        ea(value=1.02, model=model)
        self.assertEqual(ea.best_value, 1.00)
        self.assertEqual(ea.epochs_without_improvement, 1)

        # Epoch 3: loss decreases to 0.98 -> drop of 0.02 is less than min_delta 0.05 -> NOT improvement!
        ea(value=0.98, model=model)
        self.assertEqual(ea.best_value, 1.00)
        self.assertEqual(ea.epochs_without_improvement, 2)
        self.assertTrue(ea.early_stop)

    def test_higher_is_better_min_delta_logic(self):
        """Verifies that accuracy improvements respect min_delta."""
        model = nn.Linear(2, 2)
        ea = EarlyStopping(patience=2, min_delta=0.05, higher_is_better=True, verbose=False)

        # Epoch 1: initial score 0.60
        ea(value=0.60, model=model)
        self.assertEqual(ea.best_value, 0.60)

        # Epoch 2: accuracy improves to 0.70 (> 0.60 + 0.05) -> improvement!
        ea(value=0.70, model=model)
        self.assertEqual(ea.best_value, 0.70)
        self.assertEqual(ea.epochs_without_improvement, 0)

    def test_restore_best_model_weights(self):
        model = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            model.weight.fill_(1.0)

        ea = EarlyStopping(
            patience=2,
            min_delta=0.01,
            higher_is_better=False,
            restore_best_weights=True,
            verbose=False,
        )

        # Epoch 1: best loss 0.50 with weight = 1.0
        ea(value=0.50, model=model)

        # Epoch 2: worse loss with weight = 2.0
        with torch.no_grad():
            model.weight.fill_(2.0)
        ea(value=0.80, model=model)

        # Epoch 3: worse loss with weight = 3.0 -> triggers early stop
        with torch.no_grad():
            model.weight.fill_(3.0)
        ea(value=0.90, model=model)

        self.assertTrue(ea.early_stop)
        # Weights should be restored to epoch 1 (1.0)
        self.assertTrue(torch.allclose(model.weight, torch.ones_like(model.weight)))


class TestTrainingIntegration(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.in_features = 8
        self.embed_dim = 4
        self.num_classes = 2

        self.embed_net = DummyEmbeddingNet(in_features=self.in_features, embed_dim=self.embed_dim)
        self.cls_net = DummyClassifierNet(embed_dim=self.embed_dim, num_classes=self.num_classes)
        self.model = DistillationModel(
            embedding_model=self.embed_net, classifier_model=self.cls_net
        )

        # 20 samples
        x = torch.randn(20, self.in_features)
        e = torch.randn(20, self.embed_dim)
        y = torch.randint(0, self.num_classes, (20,))

        dataset = TensorDataset(x, e, y)
        self.train_loader = DataLoader(dataset, batch_size=4)
        self.val_loader = DataLoader(dataset, batch_size=4)

    def test_train_distillation_model_pipeline(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
        ea = EarlyStopping(patience=5, restore_best_weights=True, verbose=False)
        criterion = DistillationCriterion(lambda_e=1.0, lambda_y=1.0)

        history, best_epoch = train_distillation_model(
            model=self.model,
            ea=ea,
            max_epochs=3,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            optimizer=optimizer,
            criterion=criterion,
            num_classes=self.num_classes,
            lr_scheduler=scheduler,
            max_grad_norm=1.0,
            device="cpu",
        )

        self.assertEqual(history.n_epochs, 3)
        self.assertEqual(len(history.train_losses), 3)
        self.assertEqual(len(history.val_losses), 3)
        self.assertGreaterEqual(best_epoch, 1)

    def test_evaluate_distillation_model(self):
        metrics = evaluate_distillation_model(
            model=self.model,
            test_loader=self.val_loader,
            num_classes=self.num_classes,
            device="cpu",
        )
        self.assertIsInstance(metrics, ClassificationMetrics)
        res = metrics.compute()
        self.assertGreaterEqual(res.accuracy, 0.0)
        self.assertLessEqual(res.accuracy, 1.0)


if __name__ == "__main__":
    unittest.main()
