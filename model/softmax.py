import torch as t
from torch import Tensor

def stable_softmax(x: Tensor, dim: int) -> Tensor:
    #subtract max over dimension for numerical stability...
    x = x - x.max(dim=dim, keepdim=True).values

    #... and then actually calculate the softmax
    exp_x = t.exp(x)
    return exp_x / exp_x.sum(dim=dim, keepdim=True)
