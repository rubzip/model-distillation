import torch

from .distillation_model import DistillationModel


def _prepare_dummy_input(
    input_shape: tuple[int, ...], batch_size: int, device: torch.device
) -> torch.Tensor:
    """Helper to construct dummy input tensor with specified batch size."""
    if len(input_shape) >= 2 and input_shape[0] == 1:
        sample_shape = input_shape[1:]
    else:
        sample_shape = input_shape
    return torch.randn(batch_size, *sample_shape, device=device)


def _validate_embedding(e: torch.Tensor, batch_size: int, embedding_dim: int) -> None:
    if not isinstance(e, torch.Tensor):
        raise TypeError(f"Model embedding output must be a torch.Tensor, got {type(e).__name__}.")

    if e.dim() != 2:
        raise ValueError(
            f"Embedding output must be 2D [batch_size, embedding_dim], but got shape {tuple(e.shape)}."
        )

    if e.shape[0] != batch_size:
        raise ValueError(
            f"Batch size mismatch in embeddings: expected {batch_size}, got {e.shape[0]}."
        )

    if e.shape[1] != embedding_dim:
        raise ValueError(
            f"Embedding dimension mismatch: expected {embedding_dim}, got {e.shape[1]}."
        )

    if torch.isnan(e).any() or torch.isinf(e).any():
        raise ValueError("Embedding output contains NaN or Inf values.")


def _validate_classifier_output(y: torch.Tensor, batch_size: int, output_dim: int) -> None:
    if not isinstance(y, torch.Tensor):
        raise TypeError(f"Classifier output must be a torch.Tensor, got {type(y).__name__}.")

    if y.dim() != 2:
        raise ValueError(
            f"Classifier output must be 2D [batch_size, output_dim], but got shape {tuple(y.shape)}."
        )

    if y.shape[0] != batch_size:
        raise ValueError(
            f"Batch size mismatch in classification: expected {batch_size}, got {y.shape[0]}."
        )

    if y.shape[1] != output_dim:
        raise ValueError(
            f"Output class dimension mismatch: expected {output_dim}, got {y.shape[1]}."
        )

    if torch.isnan(y).any() or torch.isinf(y).any():
        raise ValueError("Classifier output contains NaN or Inf values.")


def _check_classifier_layers(classifier: torch.nn.Module) -> None:
    """Check if the classifier's final layer is Softmax or LogSoftmax."""

    def _get_last_submodule(module: torch.nn.Module) -> torch.nn.Module:
        children = list(module.children())
        if not children:
            return module
        return _get_last_submodule(children[-1])

    last_layer = _get_last_submodule(classifier)
    if isinstance(last_layer, (torch.nn.Softmax, torch.nn.LogSoftmax, torch.nn.Softmax2d)):
        raise ValueError(
            f"The classifier's final layer is {type(last_layer).__name__}. "
            "When expected_logits=True, the last layer must NOT be Softmax or LogSoftmax "
            "because nn.CrossEntropyLoss expects raw unnormalized logits."
        )


def _check_logits(y: torch.Tensor, expected_logits: bool, output_dim: int) -> None:
    if expected_logits and output_dim > 1:
        row_sums = y.sum(dim=-1)
        all_unit_sum = torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-3)
        all_non_negative = (y >= 0.0).all()
        all_below_one = (y <= 1.0).all()
        if all_unit_sum and all_non_negative and all_below_one:
            raise ValueError(
                "Classifier output appears to be softmax probabilities (sums to 1.0 and in [0, 1]). "
                "When expected_logits=True, output must be unnormalized raw logits for CrossEntropyLoss."
            )

        exp_row_sums = torch.exp(y).sum(dim=-1)
        all_exp_unit_sum = torch.allclose(exp_row_sums, torch.ones_like(exp_row_sums), atol=1e-3)
        all_non_positive = (y <= 0.0).all()
        if all_exp_unit_sum and all_non_positive:
            raise ValueError(
                "Classifier output appears to be log-softmax probabilities (exp sums to 1.0 and values <= 0). "
                "When expected_logits=True, output must be unnormalized raw logits for CrossEntropyLoss."
            )


def validate_shapes(
    model: DistillationModel,
    input_shape: tuple[int, ...],
    embedding_dim: int,
    device: str | torch.device = "cpu",
) -> bool:
    dev = torch.device(device)
    model = model.to(dev).eval()
    dummy_x = _prepare_dummy_input(input_shape, batch_size=1, device=dev)
    with torch.no_grad():
        try:
            e, _ = model(dummy_x)
        except Exception as exc:
            raise RuntimeError(
                f"Forward pass failed with input_shape={tuple(dummy_x.shape)}: {exc}"
            ) from exc

    _validate_embedding(e=e, batch_size=1, embedding_dim=embedding_dim)
    return True


def validate_output(
    model: DistillationModel,
    input_shape: tuple[int, ...],
    output_dim: int,
    expected_logits: bool = True,
    device: str | torch.device = "cpu",
) -> bool:
    dev = torch.device(device)
    model = model.to(dev).eval()
    dummy_x = _prepare_dummy_input(input_shape, batch_size=1, device=dev)
    with torch.no_grad():
        try:
            _, y = model(dummy_x)
        except Exception as exc:
            raise RuntimeError(
                f"Forward pass failed with input_shape={tuple(dummy_x.shape)}: {exc}"
            ) from exc

    _validate_classifier_output(y=y, batch_size=1, output_dim=output_dim)
    _check_logits(y=y, expected_logits=expected_logits, output_dim=output_dim)
    return True


def validate_model(
    model: DistillationModel,
    input_shape: tuple[int, ...],
    embedding_dim: int,
    output_dim: int,
    expected_logits: bool = True,
    device: str | torch.device = "cpu",
) -> bool:
    dev = torch.device(device)
    model = model.to(dev).eval()

    if expected_logits:
        _check_classifier_layers(model.classifier)

    for batch_size in (1, 2, 4, 8):
        dummy_x = _prepare_dummy_input(input_shape, batch_size, dev)
        with torch.no_grad():
            try:
                e, y = model(dummy_x)
            except Exception as exc:
                raise RuntimeError(
                    f"Forward pass failed with batch_size={batch_size} and input_shape={tuple(dummy_x.shape)}: {exc}"
                ) from exc

        _validate_embedding(e=e, batch_size=batch_size, embedding_dim=embedding_dim)
        _validate_classifier_output(y=y, batch_size=batch_size, output_dim=output_dim)
        _check_logits(y=y, expected_logits=expected_logits, output_dim=output_dim)

        preds = model.classify(dummy_x)
        if not isinstance(preds, torch.Tensor) or preds.shape != (batch_size,):
            raise ValueError(
                f"model.classify output must be a 1D Tensor with shape ({batch_size},), got {tuple(preds.shape) if isinstance(preds, torch.Tensor) else type(preds)}."
            )

    return True


__all__ = ["validate_model", "validate_shapes", "validate_output"]
