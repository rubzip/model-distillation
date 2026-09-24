from dataclasses import dataclass
import torch

@dataclass
class Metrics:
    accuracy: float
    precision: float
    recall: float
    f1: float

    def __str__(self):
        return f"Acc: {self.accuracy:.4f} | Prec: {self.precision:.4f} | Recall: {self.recall:.4f} | F1: {self.f1:.4f}"

class History:
    def __init__(self):
        self.n_epochs = 0
        self.train_losses: list[float] = []
        self.val_losses: list[float] = []
        self.results: list[Metrics] = []

    def add(self, train_loss:float, val_loss: float, result: Metrics):
        self.n_epochs += 1
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.results.append(result)

class ClassificationMetrics:
    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.conf_matrix = torch.zeros((self.num_classes, self.num_classes), dtype=torch.int64)

    @torch.no_grad()
    def update(self, y_pred_logits: torch.Tensor, y_true: torch.Tensor):
        preds = torch.argmax(y_pred_logits, dim=-1)
        indices = self.num_classes * y_true.view(-1) + preds.view(-1)
        bincount = torch.bincount(indices, minlength=self.num_classes ** 2)
        self.conf_matrix += bincount.reshape((self.num_classes, self.num_classes)).cpu()

    def compute(self) -> Metrics:
        tp = self.conf_matrix.diagonal()
        
        pred_counts = self.conf_matrix.sum(dim=0)
        true_counts = self.conf_matrix.sum(dim=1)
        
        precision = torch.where(pred_counts > 0, tp / pred_counts, 0.0)
        recall = torch.where(true_counts > 0, tp / true_counts, 0.0)
        
        p_plus_r = precision + recall
        f1 = torch.where(p_plus_r > 0, 2 * (precision * recall) / p_plus_r, 0.0)
        
        total_samples = self.conf_matrix.sum().item()
        accuracy = tp.sum().item() / total_samples if total_samples > 0 else 0.0
        
        return Metrics(
            accuracy=accuracy,
            precision=precision.mean().item(),
            recall=recall.mean().item(),
            f1=f1.mean().item()
        )

    def get_cm(self, normalized: bool = True) -> torch.Tensor:
        if not normalized:
            return self.conf_matrix.clone()
        
        cm_float = self.conf_matrix.float()
        row_sums = cm_float.sum(dim=1, keepdim=True)
        cm_normalized = torch.where(row_sums > 0, cm_float / row_sums, 0.0)        
        return cm_normalized
