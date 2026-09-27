import torch

from ..models import DistillationModel
from .criterion import AbstractCriterion
from .metrics import ClassificationMetrics


def evaluate_distillation_model(
    model: DistillationModel,
    test_loader: torch.utils.data.DataLoader,
    num_classes: int,
    device: str | torch.device = "cpu",
    criterion: AbstractCriterion | None = None,
) -> ClassificationMetrics | tuple[ClassificationMetrics, float]:
    """Evaluates a distillation model on test data, computing classification metrics and optional loss."""
    dev = torch.device(device)
    metrics = ClassificationMetrics(num_classes=num_classes)
    model.to(dev)
    model.eval()

    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in test_loader:
            if len(batch) == 3:
                x, e, y = batch
                e = e.to(dev)
            else:
                x, y = batch
                e = None

            x, y = x.to(dev), y.to(dev)
            batch_size = x.size(0)

            e_pred, y_pred = model(x)
            metrics.update(y_pred_logits=y_pred, y_true=y)

            if criterion is not None and e is not None:
                loss_out = criterion(e_true=e, e_pred=e_pred, y_true=y, y_pred=y_pred)
                loss = loss_out[0] if isinstance(loss_out, tuple) else loss_out
                total_loss += loss.item() * batch_size
                total_samples += batch_size

    if criterion is not None and total_samples > 0:
        return metrics, total_loss / total_samples

    return metrics


__all__ = ["evaluate_distillation_model"]
