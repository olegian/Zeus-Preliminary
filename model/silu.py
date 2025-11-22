import torch

def silu(x: torch.Tensor) -> torch.Tensor:
    """Swish function, expects x to be 1xN"""
    return x * torch.sigmoid(x)