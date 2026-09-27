from abc import ABC, abstractmethod

import torch


class BaseProcessing(ABC):
    def __init__(self, config: dict):
        self.config = config

    @abstractmethod
    def process(self, data: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def to(self, device: torch.device) -> None:
        pass

    def eval(self):
        if isinstance(self, torch.nn.Module):
            self.eval()
    
    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        return self.process(data)
