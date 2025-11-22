import functools
from typing import List

import torch


def clip_gradient(
    params: List[torch.nn.Parameter],
    max_grad_l2_norm: float,
    eps: float =1e-5,
):
    # filter once and then reuse later
    params_with_grad = list(filter(lambda param: param.grad is not None, params))

    acc_grad = functools.reduce(
        lambda acc, param: acc + (param.grad.data.norm(2).item() ** 2),
        params_with_grad,
        0
    )
    total = (acc_grad + eps) ** 0.5

    if total > max_grad_l2_norm:
        for param in params_with_grad:
            param.grad.data.mul_(max_grad_l2_norm / (total + eps))