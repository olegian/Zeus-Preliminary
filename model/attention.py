from jaxtyping import Bool, Float
import torch as t
from torch import Tensor
import torch

from model.softmax import stable_softmax

def attention(
        Q: Float[Tensor, " ... queries d_k"],
        K: Float[Tensor, " ... keys d_k"],
        V: Float[Tensor, " ... values d_v"],
        mask: Bool[Tensor, " ... queries keys"] | None = None,
    ) -> Float[Tensor, " ... queries d_v"]:
    
    d_k = Q.shape[-1]

    scale = d_k ** 0.5
    scores: torch.Tensor = (Q @ K.transpose(-1, -2)) / scale

    if mask is not None:
        mask = mask.to(dtype=torch.bool, device=scores.device)
        scores = scores.masked_fill(~mask, float('-inf'))

    attn_weights = stable_softmax(scores, dim=-1)
    return attn_weights @ V