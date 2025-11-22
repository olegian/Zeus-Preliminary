import os
import typing
import torch

def save_checkpoint(
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        iteration: int,
        out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]
    ):
    state = {
        "model": model.state_dict(),
        "opt": optimizer.state_dict(),
        "iter": iteration,
    }
    torch.save(state, out)

def load_checkpoint(
        src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> int:
    state = torch.load(src)

    model.load_state_dict(state['model'])
    optimizer.load_state_dict(state['opt'])

    return state['iter']