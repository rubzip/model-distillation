import torch
import torch.nn as nn
from abc import ABC, abstractmethod

class AbstractCriterion(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(
        self, 
        e_true: torch.Tensor, 
        e_pred: torch.Tensor, 
        y_true: torch.Tensor, 
        y_pred: torch.Tensor
    ) -> torch.Tensor:
        pass

class DistillationCriterion(AbstractCriterion):
    def __init__(self, lambda_e: float, lambda_y: float, loss_e_fn = None, loss_y_fn = None):
        super().__init__()
        self.lambda_e = lambda_e
        self.lambda_y = lambda_y
        self.loss_e_fn = nn.MSELoss() if loss_e_fn is None else loss_e_fn
        self.loss_y_fn = nn.CrossEntropyLoss() if loss_y_fn is None else loss_y_fn

    def forward(self, e_true, e_pred, y_true, y_pred):
        loss_e = self.loss_e_fn(e_pred, e_true)
        loss_y = self.loss_y_fn(y_pred, y_true)
        return self.lambda_y * loss_y + self.lambda_e * loss_e
