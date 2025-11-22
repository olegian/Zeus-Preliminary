from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import torch
import torch.distributed as dist

MB_TO_BYTES = 1024 * 1024

class Bucket:
    def __init__(self):
        self.parameters: List[torch.nn.Parameter] = []
        self.ready_to_send: int = 0
        self.byte_size: int = 0
    
    def add(self, param: torch.nn.Parameter):
        self.parameters.append(param)
        self.byte_size += param.numel() * param.element_size()
    
    def prepare(self) -> bool:
        self.ready_to_send += 1
        return self.ready_to_send == len(self.parameters)
    
    def flatten_grads(self) -> torch.Tensor:
        self.flat_grads = torch.cat([param.grad.view(-1) for param in self.parameters])
        return self.flat_grads

    def __str__(self):
        return str([param.grad for param in self.parameters])

class DDPBucketedParameters(torch.nn.Module):
    def __init__(self, module: torch.nn.Module, bucket_size_mb: Optional[float] = None):
        """Given an instantiated PyTorch nn.Module to be parallelized,
        constructs a DDP container that will handle gradient synchronization
        across ranks.
        """
        super().__init__()
        self.module = module
        # buckets, param -> bucket_idx into self.buckets for bucket which owns this parameter
        self.buckets, self.param_to_bucket_idx = self._build_bounded_buckets(bucket_size_mb)
        # handle -> bucket_idx into self.bucket for bucket which this handle is responsible for
        self.comm_handles: Dict[dist.Work, int] = {}

        for param in self.module.parameters():
            dist.broadcast(param.data, 0)

        def comm_hook(param: torch.Tensor):
            if param.requires_grad:
                # determine which bucket this param belongs to
                b_idx = self.param_to_bucket_idx[param]
                bucket = self.buckets[b_idx]
                if bucket.prepare(): # mark param in this bucket as ready, if
                    # all the gradients of this bucket have been computed, 
                    # we can all-reduce!
                    flat_grads = bucket.flatten_grads()
                    handle = dist.all_reduce(flat_grads, dist.ReduceOp.SUM, async_op=True)
                    self.comm_handles[handle] = b_idx 

        for param in self.module.parameters():
            if param.requires_grad:
                param.register_post_accumulate_grad_hook(comm_hook)

    def forward(self, *inputs, **kwargs):
        """Calls the wrapped module’s forward() method with the provided
        positional and keyword arguments.
        """ 
        return self.module(*inputs, **kwargs)

    def _build_bounded_buckets(self, megabyte_limit: Optional[float] = None):
        """Stores references to bucketed gradients, in reverse order"""
        # reinit just in case this is ever called twice (although that should never happen)
        buckets: List[Bucket] = []
        param_to_bucket_idx: Dict[torch.nn.Parameter, int] = {}
        current_bucket = Bucket()
        for param in reversed(list(self.module.parameters())):
            if not param.requires_grad:
                continue
        
            grad_size = param.numel() * param.element_size()
            if megabyte_limit is not None and current_bucket.byte_size + grad_size >= megabyte_limit * MB_TO_BYTES:
                # if adding this gradient to the current bucket would exceed the limit
                # then we are done building up this bucket, so register this bucket
                # as complete, and reset current state
                buckets.append(current_bucket)
                current_bucket = Bucket()

            current_bucket.add(param)
            param_to_bucket_idx[param] = len(buckets)
        
        # last current_bucket could be smaller than byte limit, if 
        # we need to, add the smaller bucket as the last registered bucket
        if current_bucket:
            buckets.append(current_bucket)

        return buckets, param_to_bucket_idx

    def finish_gradient_synchronization(self): 
        """When called, wait for asynchronous communication calls to be
        queued on the GPU
        """
        for handle in self.comm_handles.keys():
            handle.wait()            
            bucket: Bucket = self.buckets[self.comm_handles[handle]]
            bucket.flat_grads /= dist.get_world_size()
            
            offset = 0
            for param in bucket.parameters:
                param.grad.copy_(bucket.flat_grads[offset:offset + param.grad.numel()].view_as(param.grad))
                offset += param.grad.numel()

        self.comm_handles.clear()
        for bucket in self.buckets:
            bucket.ready_to_send = 0