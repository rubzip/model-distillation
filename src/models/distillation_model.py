import torch
import torch.nn as nn


class DistillationModel(nn.Module):
    """Composite model combining an embedding extractor and a classification head for knowledge distillation."""

    def __init__(self, embedding_model: nn.Module, classifier_model: nn.Module):
        super().__init__()
        if not isinstance(embedding_model, nn.Module):
            raise TypeError(
                f"embedding_model must be an instance of nn.Module, got {type(embedding_model).__name__}."
            )
        if not isinstance(classifier_model, nn.Module):
            raise TypeError(
                f"classifier_model must be an instance of nn.Module, got {type(classifier_model).__name__}."
            )
        self.embedding = embedding_model
        self.classifier = classifier_model

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass computing both intermediate embeddings and output logits."""
        e = self.embedding(x)
        y = self.classifier(e)
        return e, y

    @torch.no_grad()
    def classify(self, x: torch.Tensor) -> torch.Tensor:
        """Predict class indices for input samples using argmax over classifier outputs."""
        _, y = self(x)
        c = torch.argmax(y, dim=-1)
        return c
