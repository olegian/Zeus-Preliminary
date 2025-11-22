
from typing import Callable, Optional, Tuple
import torch

class AdamWOptimizer(torch.optim.Optimizer):
    def __init__(self, 
            params, 
            lr: float = 0.001, 
            betas: Tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-7, 
            weight_decay: float = 0.1
        ):
        #just in case lol
        assert lr > 0
        assert eps > 0
        assert weight_decay > 0
        assert betas[0] > 0
        assert betas[1] > 0


        super().__init__(params, defaults={
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        })

    def step(self, closure: Optional[Callable] = None):
        # why does the optimizer use this closure format for loss?
        loss = None if closure is None else closure()
        
        for group in self.param_groups:
            for param in group['params']:
                if param.grad is None:
                    continue

                param_state = self.state[param]

                # numbers match up to spec algorithm on part 1 page 22
                # 1: init(theta)
                # 2: m <- 0 (first moment)
                # 3: v <- 0 (second moment)
                #    t <- 1 (number of steps for lr adjustment)
                # only do this if it hasnt been initialized already
                # im scared that this is a bad check, but its been working so far
                #!if i start getting bad values i gotta check the docs for this, or just swap to check for t, m, v membership
                if len(param_state) == 0:
                    param_state['t'] = 0 #increased to 1 after this init
                    param_state['m'] = torch.zeros_like(param.data)
                    param_state['v'] = torch.zeros_like(param.data)
                
                m: torch.Tensor = param_state['m']
                v: torch.Tensor = param_state['v']
                beta1, beta2 = group['betas']

                param_state['t'] += 1 #doing this here also means it gets inited to 1

                # 6: grad <- grad of tensor
                grad = param.grad.data

                # 7: m <- beta1 * m + (1 - beta1) * grad
                # 8: v <- beta2 * v + (1 - beta2) * grad^2
                # sticking with inplace ops with func_
                m.mul_(beta1).add_(grad, alpha=(1 - beta1))
                v.mul_(beta2).addcmul_(grad, grad, value=(1 - beta2))

                # 9: equations long, so in english compute scale factor via b1 and b2, then update lr
                step_num = param_state['t']
                adjustment = ((1 - beta2 ** step_num) ** 0.5) / (1 - beta1 ** step_num)
                lr_t = group['lr'] * adjustment

                # 10: theta <- theta - (lr_t) * (m / (root(v) + eps))
                # actually take step to update param (again, using inplace ops func_)
                # addcdiv_ here does m / (root(v) + eps), then scales by `value` which is -lr
                param.data.addcdiv_(m, v.sqrt().add_(group['eps']), value=-lr_t)
                
                # 11: theta <- theta - (lr)(lambda_)(theta)
                # applying weight decay. note the lr here is the unadjusted one
                param.data.add_(param.data, alpha=-group["lr"] * group['weight_decay'])

        return loss
