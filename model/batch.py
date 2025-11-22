from typing import Tuple
import numpy as np
import torch


def data_loader(
        token_ids: np.ndarray, 
        batch_size: int,
        context_length: int,
        device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:

    rng = np.random.default_rng()

    end = len(token_ids) - context_length
    window_starts = rng.integers(0, end, size=batch_size)

    return torch.tensor(np.stack([token_ids[(start):(start+context_length)] for start in window_starts]), dtype=torch.long, device=device), \
           torch.tensor(np.stack([token_ids[(start+1):(start + context_length + 1)] for start in window_starts]), dtype=torch.long, device=device)
