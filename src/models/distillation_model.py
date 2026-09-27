import torch
import torch.nn as nn


class DistillationModel(nn.Module):
    def __init__(self, embedding_model: nn.Module, classifier_model: nn.Module):
        super().__init__()
        self.embedding = embedding_model
        self.classifier = classifier_model

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        e = self.embedding(x)
        y = self.classifier(e)
        return e, y

    @torch.no_grad()
    def classify(self, x: torch.Tensor) -> torch.Tensor:
        _, y = self(x)
        c = torch.argmax(y, dim=-1)
        return c
