from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math

def get_lr_cosine_schedule(
        it: int,
        max_lr: float,
        min_lr: float,
        warmup_iters: int,
        cosine_cycle_iters: int
):
    if it < warmup_iters:
        return it * max_lr / warmup_iters
    elif warmup_iters <= it <= cosine_cycle_iters:
        cosine_val = math.cos(math.pi * (it - warmup_iters) / (cosine_cycle_iters - warmup_iters))
        return min_lr + (1 + cosine_val) * (max_lr - min_lr) / 2
    else:
        return min_lr


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01):
        if lr < 0:
            raise ValueError("Invaid learning rate value")
        if eps < 0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= beta1 < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {beta1}")
        if not 0.0 <= beta2 < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {beta2}")
        
        defaults = {
            "lr": lr,
            "eps": eps,
            "beta1": beta1,
            "beta2": beta2,
            "weight_decay": weight_decay
        }

        super.__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            eps = group["eps"]
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                
                state = self.state[p]
                t = state.get("t", 0)
                m = state.get("m", 0)
                v = state.get("v", 0)
                grad = p.grad.data
                cur_lr = lr * math.sqrt(1 - beta2 ** 2) / (1 - beta1 ** 2)
                p.data -= (lr * weight_decay * p.data)
                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * (grad ** 2)
                p.data -= (cur_lr * m / (math.sqrt(v) + eps))
                state["t"] = t + 1
                state["m"] = m
                state["v"] = v
            
        return loss
