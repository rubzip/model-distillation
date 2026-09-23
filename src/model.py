import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, embedding: nn.Module, classifier: nn.Module):
        super().__init__()
        self.embedding = embedding
        self.classifier = classifier

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        e = self.embedding(x)
        y = self.classifier(e)
        return e, y

    @torch.no_grad()
    def classify(self, x: torch.Tensor) -> torch.Tensor:
        _, y = self(x)
        c = torch.argmax(y, dim=-1)
        return c
