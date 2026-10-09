import torch
import torch.nn as nn
from thop import profile


def get_size_MBytes(model: nn.Module) -> float:
    """Calculate the memory footprint of model parameters and buffers in megabytes (MB)."""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / (1024**2)
    return float(size_all_mb)


def get_operations_MFlop(
    model: nn.Module,
    input_shape: tuple[int, ...] = (1, 3, 224, 224),
    device: str | torch.device = "cpu",
) -> float:
    """Compute theoretical computation cost in MFLOPs for a given input shape."""
    dev = torch.device(device)
    dummy_input = torch.randn(*input_shape).to(dev)
    model = model.to(dev)
    was_training = model.training
    model.eval()

    try:
        with torch.no_grad():
            macs, _ = profile(model, inputs=(dummy_input,), verbose=False)
        flops = 2 * macs
        mflops = flops / 1e6
        return float(mflops)
    finally:
        if was_training:
            model.train()

