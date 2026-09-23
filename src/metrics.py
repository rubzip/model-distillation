import torch

class ClassificationMetrics:
    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        # Matriz de confusión C x C
        self.conf_matrix = torch.zeros((self.num_classes, self.num_classes), dtype=torch.int64)

    @torch.no_grad()
    def update(self, y_pred_logits: torch.Tensor, y_true: torch.Tensor):
        preds = torch.argmax(y_pred_logits, dim=-1)
        indices = self.num_classes * y_true.view(-1) + preds.view(-1)
        bincount = torch.bincount(indices, minlength=self.num_classes ** 2)
        self.conf_matrix += bincount.reshape((self.num_classes, self.num_classes)).cpu()

    def compute(self) -> dict:
        tp = self.conf_matrix.diagonal()
        
        pred_counts = self.conf_matrix.sum(dim=0)
        true_counts = self.conf_matrix.sum(dim=1)
        
        precision = torch.where(pred_counts > 0, tp / pred_counts, 0.0)
        recall = torch.where(true_counts > 0, tp / true_counts, 0.0)
        
        p_plus_r = precision + recall
        f1 = torch.where(p_plus_r > 0, 2 * (precision * recall) / p_plus_r, 0.0)
        
        return {
            "accuracy": tp.sum().item() / self.conf_matrix.sum().item(),
            "precision": precision.mean().item(),
            "recall": recall.mean().item(),
            "f1": f1.mean().item()
        }
