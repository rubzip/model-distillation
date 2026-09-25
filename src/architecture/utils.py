import torch
import torch.nn as nn
from thop import profile


def get_size_MBytes(model: nn.Module) -> float:
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / (1024**2)
    return size_all_mb


def get_operations_MFlop(
    model: nn.Module, input_shape: tuple = (1, 3, 224, 224), device: str = "cpu"
) -> float:
    dummy_input = torch.randn(*input_shape).to(device)
    model = model.to(device)

    macs, params = profile(model, inputs=(dummy_input,), verbose=False)
    flops = 2 * macs
    mflops = flops / 1e6
    return mflops
