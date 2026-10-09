from torch.nn import Dropout, Linear, Module, ReLU, Sequential


class MLPFactory:
    """Factory for creating multi-layer perceptron (MLP) classification heads."""

    @staticmethod
    def create_mlp(
        input_size: int,
        hidden_sizes: list[int],
        num_classes: int,
        dropout_rate: float = 0.5,
    ) -> Module:
        """Create a Sequential MLP classifier outputting raw unnormalized logits."""
        if input_size <= 0:
            raise ValueError(f"input_size must be positive, got {input_size}")
        if num_classes <= 0:
            raise ValueError(f"num_classes must be positive, got {num_classes}")
        if not (0.0 <= dropout_rate < 1.0):
            raise ValueError(f"dropout_rate must be in [0.0, 1.0), got {dropout_rate}")

        layers: list[Module] = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            if hidden_size <= 0:
                raise ValueError(f"Hidden layer size must be positive, got {hidden_size}")
            layers.append(Linear(prev_size, hidden_size))
            layers.append(ReLU())
            if dropout_rate > 0:
                layers.append(Dropout(dropout_rate))
            prev_size = hidden_size

        layers.append(Linear(prev_size, num_classes))
        return Sequential(*layers)

