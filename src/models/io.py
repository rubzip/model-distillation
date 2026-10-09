from pathlib import Path

import torch

from .distillation_model import DistillationModel


def save_distillation_model(model: DistillationModel, path: str | Path) -> None:
    """Save the distillation model state dictionary to disk, creating parent directories if needed."""
    save_path = Path(path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)


def load_distillation_model(
    model: DistillationModel,
    path: str | Path,
    device: str | torch.device = "cpu",
) -> DistillationModel:
    """Load weights into a distillation model from disk and map to the specified device."""
    dev = torch.device(device)
    model.load_state_dict(torch.load(path, map_location=dev, weights_only=True))
    model.to(dev)
    return model

