import math

import torch
import torch.nn as nn


class DumbEmbedding(nn.Module):
    def __init__(self, input_shape: tuple[int, ...], embedding_dim: int):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(math.prod(input_shape), embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        x = self.linear(x)
        return x
