import torch

from src.processing.base import BaseProcessing


class VGGishProcessor(BaseProcessing):
    def __init__(self, target_dim: int = 128):
        super().__init__({})
        self.target_dim = target_dim

    def process(self, data: torch.Tensor) -> torch.Tensor:
        # Stub implementation
        return torch.randn((data.size(0), self.target_dim), device=data.device)

    def to(self, device: torch.device) -> 'VGGishProcessor':
        return self
