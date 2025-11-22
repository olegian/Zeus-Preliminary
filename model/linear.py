import torch
import torch.nn as nn

class LinearLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        """ Construct a linear transformation module.
        This function should accept the following parameters:
            in_features: int
                Final dimension of the input
            out_features: int
                Final dimension of the output
            device: torch.device | None = None
                Device to store the parameters on
            dtype: torch.dtype | None = None
                Data type of the parameters
        """
        super().__init__()

        self.w = nn.Parameter(
            torch.zeros(out_features, in_features, device=device, dtype=dtype)
        )
        nn.init.trunc_normal_(self.w)  # init with truncated normal distribution


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Apply the linear transformation to the input. """
        return x @ self.w.T
