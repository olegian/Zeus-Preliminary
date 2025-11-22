from typing import Tuple
import torch
import torch.distributed as dist

class DDPIndividualParameters(torch.nn.Module):
    def __init__(self, module: torch.nn.Module):
        """Given an instantiated PyTorch nn.Module to be parallelized,
        constructs a DDP container that will handle gradient synchronization
        across ranks.
        """
        super().__init__()
        self.module = module
        self.comm_handles = []

        for param in self.module.parameters():
            dist.broadcast(param.data, 0)

        def comm_hook(param: torch.Tensor):
            if param.grad is not None:
                handle = dist.all_reduce(param.grad, dist.ReduceOp.SUM, async_op=True)
                self.comm_handles.append(handle)

        # self.module.register_full_backward_hook(lambda *args, **kwargs: self.comm_hook(*args, **kwargs))
        for param in self.module.parameters():
            if param.requires_grad:
                param.register_post_accumulate_grad_hook(comm_hook)

    def forward(self, *inputs, **kwargs):
        """Calls the wrapped module’s forward() method with the provided
        positional and keyword arguments.
        """ 
        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self): 
        """When called, wait for asynchronous communication calls to be
        queued on the GPU
        """
        for handle in self.comm_handles:
            handle.wait()            
        self.comm_handles.clear()

        for param in self.module.parameters():
            if param.grad is not None:
                param.grad /= dist.get_world_size()