import torch
import torch.nn as nn

from model.silu import silu

class SwiGluLayer(nn.Module):
    def __init__(
        self, 
        d_model: int, 
        d_ff: int, 
        device=None, dtype=None
    ):
        """ Construct the SwiGLU module.
        This function should accept the following parameters:
            d_model: Dim of the feedforward input / output
            d_ff: Dim of the up-projection
            device: torch.device | None - Device to store the parameters on
            dtype: torch.dtype | None - Data type of the parameters
        """
        super().__init__()

        # d_ff d_model
        self.w1 = nn.Parameter(
            torch.zeros(d_ff, d_model, device=device, dtype=dtype)
        )
        nn.init.trunc_normal_(self.w1)

        # d_model d_ff
        self.w2 = nn.Parameter(
            torch.zeros(d_model, d_ff, device=device, dtype=dtype)
        )
        nn.init.trunc_normal_(self.w2)

        # d_ff d_model
        self.w3 = nn.Parameter(
            torch.zeros(d_ff, d_model, device=device, dtype=dtype)
        )
        nn.init.trunc_normal_(self.w3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Process input tensor of shape (batch_size, sequence_length, d_model)
        returning a tensor of the same shape """

        # todo: combine x @ self.w1 and x @ self.w3 via concat / split
        return ( silu(x @ self.w1.T) * (x @ self.w3.T) ) @ self.w2.T
