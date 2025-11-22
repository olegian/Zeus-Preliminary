import einops
import torch
import torch.nn as nn

from model.attention import attention
from model.rotary_position_embedding import RotaryPositionalEmbeddingLayer

class CausalMultiHeadAttentionLayer(nn.Module):
    def __init__(self, 
        d_model: int, 
        num_heads: int,
        mask: bool = False,
        theta: float | None = None,
        max_seq_len: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None):
        """ Construct a module to perform causal multi-head attention
        This function accepts the following parameters:
            d_model: int
                Dimensionality of the Transformer block inputs
            num_heads: int
                Number of heads to use in multi-head self-attention
            device: torch.device | None = None
                Device to store the parameters on
            dtype: torch.dtype | None = None
                Data type of the parameters
        """
        super().__init__()

        self.d_model = d_model
        self.n_heads = num_heads
        self.d_kv = d_model // num_heads
        self.mask = mask

        self.W = nn.Parameter(torch.empty((d_model, 3 * d_model), device=device, dtype=dtype))
        nn.init.trunc_normal_(self.W)

        self.W_o = nn.Parameter(torch.empty((d_model, d_model), device=device, dtype=dtype))
        nn.init.trunc_normal_(self.W_o)

        # todo: make a parsed type for rope parameters to not do this ugly check
        self.rope = RotaryPositionalEmbeddingLayer(theta, self.d_kv, max_seq_len=max_seq_len, device=device) if theta is not None and max_seq_len is not None else None

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        """ Run the batched input tensor through the layer
        Args:
            x (torch.Tensor): shape [...batches, seq_len, d_model]

        Returns: torch.Tensor
        """

        w = einops.rearrange(self.W, "d_model (three d_model_qkv) -> d_model three d_model_qkv", three=3)
        proj = einops.einsum(x, w, '... seq qkv, d_model three qkv -> ... seq three d_model')
        split: torch.Tensor = einops.rearrange(proj, '... seq three (n_heads d_k) -> ... three n_heads seq d_k', n_heads=self.n_heads)
        q, k, v = split.unbind(-4)

        if self.rope is not None and token_positions is not None:
            q = self.rope.forward(q, token_positions)
            k = self.rope.forward(k, token_positions)

        mask = ~torch.triu(torch.ones((x.shape[-2], x.shape[-2]), device=x.device, dtype=torch.bool), diagonal=1) if self.mask else None
        attn = attention(q, k, v, mask) 

        concat = einops.rearrange(attn, '... h seq d_v -> ... seq (h d_v)')
        return einops.einsum(concat, self.W_o, '... seq d_model, out d_model -> ... seq out')
