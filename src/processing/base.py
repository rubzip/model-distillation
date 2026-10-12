from abc import ABC, abstractmethod

import torch


class BaseProcessing(ABC):
    def __init__(self, config: dict):
        self.config = config

    @abstractmethod
    def process(self, data: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def to(self, device: torch.device) -> "BaseProcessing":
        pass

    def eval(self):
        if hasattr(super(), "eval"):
            super().eval()
        for attr_name in dir(self):
            try:
                attr = getattr(self, attr_name)
                if isinstance(attr, torch.nn.Module):
                    attr.eval()
            except Exception:
                pass

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        return self.process(data)
