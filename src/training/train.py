import torch

from ..models.distillation_model import DistillationModel
from .criterion import AbstractCriterion
from .early_stopping import EarlyStopping
from .metrics import ClassificationMetrics, History


def train_distillation_model(
    model: DistillationModel,
    ea: EarlyStopping,
    max_epochs: int,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: AbstractCriterion,
    num_classes: int,
    device: str | torch.device = "cpu",
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    max_grad_norm: float | None = None,
) -> tuple[History, int]:
    """Standard training loop for knowledge distillation with EarlyStopping and metrics."""
    dev = torch.device(device)
    model.to(dev)
    metrics = ClassificationMetrics(num_classes=num_classes)
    history = History()

    for epoch in range(max_epochs):
        # --- Training Phase ---
        model.train()
        train_loss = 0.0
        total_train_samples = 0

        for x, e, y in train_loader:
            x, e, y = x.to(dev), e.to(dev), y.to(dev)
            batch_size = x.size(0)

            optimizer.zero_grad()
            e_pred, y_pred = model(x)
            loss = criterion(e_true=e, e_pred=e_pred, y_true=y, y_pred=y_pred)

            loss.backward()

            if max_grad_norm is not None and max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)

            optimizer.step()

            train_loss += loss.item() * batch_size
            total_train_samples += batch_size

        train_loss /= total_train_samples if total_train_samples > 0 else 1.0

        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        total_val_samples = 0
        metrics.reset()

        with torch.no_grad():
            for x, e, y in val_loader:
                x, e, y = x.to(dev), e.to(dev), y.to(dev)
                batch_size = x.size(0)

                e_pred, y_pred = model(x)
                loss = criterion(e_true=e, e_pred=e_pred, y_true=y, y_pred=y_pred)

                val_loss += loss.item() * batch_size
                total_val_samples += batch_size
                metrics.update(y_pred_logits=y_pred, y_true=y)

        val_loss /= total_val_samples if total_val_samples > 0 else 1.0

        # Step LR scheduler if provided
        if lr_scheduler is not None:
            if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                lr_scheduler.step(val_loss)
            else:
                lr_scheduler.step()

        m_results = metrics.compute()
        history.add(train_loss=train_loss, val_loss=val_loss, result=m_results)

        print(
            f"Epoch {epoch + 1}/{max_epochs} | Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | {m_results}"
        )

        ea(value=val_loss, model=model)
        if ea.early_stop:
            print("Training terminated by EarlyStopping.")
            break

    if ea.restore_best_weights and ea.best_model is not None:
        ea.restore_best_model(model)

    return history, ea.best_epoch


__all__ = ["train_distillation_model"]
