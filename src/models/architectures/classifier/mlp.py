import torch.nn as nn

# import Module, Linear, ReLU, Sequential, Dropout


class MLPFactory:
    @staticmethod
    def create_mlp(input_size: int, hidden_sizes: list[int], num_classes: int, dropout_rate: float = 0.5) -> nn.Module:
        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            if hidden_size <= 0:
                raise ValueError(f"Hidden layer size must be positive, got {hidden_size}")
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, num_classes))
        return nn.Sequential(*layers)
