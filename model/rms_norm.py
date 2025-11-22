from typing import Dict
from sympy import root
import torch
import torch.nn as nn

class RMSNormLayer(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        """ Construct the RMSNorm module.
        This function should accept the following parameters:
            d_model: int - Hidden dimension of the model
            eps: float = 1e-5 - Epsilon value for numerical stability
            device: torch.device | None - Device to store the parameters on
            dtype: torch.dtype | None - Data type of the parameters
        """
        super().__init__()

        self.eps = eps
        self.gains = nn.Parameter(
            torch.ones(d_model, device=device, dtype=dtype)
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Process an input tensor of shape (batch_size, sequence_length, d_model)
        and return a tensor of the same shape."""

        # calc RMS only over last dimension (d_model)
        root_mean_square = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return (x / root_mean_square) * self.gains  # normalize input with calculated RMS
