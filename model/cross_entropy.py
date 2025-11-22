import torch

def cross_entropy_loss(logits: torch.Tensor, targets: torch.Tensor):
    """ Args:
        logits (Float[Tensor, "batch_size vocab_size"]): logits[i][j] is the
            unnormalized logit of jth class for the ith example.
        targets (Int[Tensor, "batch_size"]): Tensor of shape (batch_size,) with the index of the correct class.
            Each value must be between 0 and `num_classes - 1`.
    """
    # numerical stablity
    logits = logits - logits.max(dim=-1, keepdim=True).values

    target_logits = logits.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    # use log properties to split fraction in softmax, cancel log/exp
    loss = torch.logsumexp(logits, dim=-1) - target_logits  # (batch) shape
    return loss.mean()