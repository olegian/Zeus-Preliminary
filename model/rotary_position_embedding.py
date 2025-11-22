import torch
import torch.nn as nn

class RotaryPositionalEmbeddingLayer(nn.Module):
    def __init__(
        self, 
        theta: float,
        d_k: int, 
        max_seq_len: int,
        device=None
    ):
        """ Construct the RotaryPositionalEmbedding module.
        This function should accept the following parameters:
            theta: Theta constant for RoPE
            d_k: Dim of q/k vectors (must be even), 
            max_seq_len: max sequence length that will be input,
            device: torch.device | None - Device to store the parameters on
        """
        super().__init__()

        # together, these construct R^i
        seq_idxs = torch.arange(max_seq_len, device=device)
        inverses = 1.0 / (theta ** (torch.arange(0, d_k, 2, device=device, dtype=torch.float32) / d_k))
        outer_product = torch.outer(seq_idxs, inverses).float().to(device)

        # freqs are shape [max_seq_len, d_k // 2]
        self.register_buffer("rope_sin", outer_product.sin(), persistent=False)
        self.register_buffer("rope_cos", outer_product.cos(), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """ Apply RoPE to an input tensor of shape (..., seq_len, d_k) and
        return a tensor of the same shape.
        Notes:
        - Accept x with an arbitrary number of batch dimensions.
        - token_positions has shape (..., seq_len) and gives absolute
          positions per token along the sequence dimension.
        - Use token_positions to slice (precomputed) cos/sin tensors
          along the sequence dimension.
        """

        evens, odds = x[..., ::2], x[..., 1::2] # even idxs of d_k, odd idxs of d_k to do pairwise rotation
        sin, cos = self.rope_sin[token_positions], self.rope_cos[token_positions]

        # apply rotation, stacking tensor back together afterwards
        return torch.stack(
          [evens * cos - odds * sin,
           evens * sin + odds * cos], dim=-1
        ).flatten(-2)
