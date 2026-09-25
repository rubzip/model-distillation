import torch

from .criterion import AbstractCriterion
from .early_stopping import EarlyStopping
from .metrics import ClassificationMetrics, History
from .model import Model


def train(
    model: Model,
    ea: EarlyStopping,
    max_epochs: int,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: AbstractCriterion,
    num_classes: int,
    device: torch.device = torch.device("cpu"),
) -> History:
    """
    Standard training loop utilizing the EarlyStopping mechanism and metrics.
    """
    metrics = ClassificationMetrics(num_classes=num_classes)
    model.to(device)
    history = History()

    for epoch in range(max_epochs):
        # --- Training Phase ---
        model.train()
        train_loss = 0.0

        for x, e, y in train_loader:
            x, e, y = x.to(device), e.to(device), y.to(device)

            optimizer.zero_grad()
            e_pred, y_pred = model(x)
            loss = criterion(e_true=e, e_pred=e_pred, y_true=y, y_pred=y_pred)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x.size(0)

        train_loss /= len(train_loader.dataset)

        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        metrics.reset()

        with torch.no_grad():
            for x, e, y in val_loader:
                x, e, y = x.to(device), e.to(device), y.to(device)

                e_pred, y_pred = model(x)
                loss = criterion(e_true=e, e_pred=e_pred, y_true=y, y_pred=y_pred)

                val_loss += loss.item() * x.size(0)
                metrics.update(y_pred_logits=y_pred, y_true=y)

        val_loss /= len(val_loader.dataset)

        m_results = metrics.compute()
        history.add(train_loss, val_loss, m_results)

        print(f"Epoch {epoch + 1}/{max_epochs} | Train Loss: {train_loss:.4f} | {m_results}")

        ea(value=val_loss, model=model)
        if ea.early_stop:
            print("Training terminated by EarlyStopping.")
            break

    return history, ea.best_epoch
