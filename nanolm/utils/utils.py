import torch
from torch import Tensor
from jaxtyping import Float, Int
from collections.abc import Iterable

def softmax(x: Tensor, dim: int) -> Tensor:
    x = x - torch.max(x, dim=dim, keepdim=True).values
    exp_x = torch.exp(x)
    exp_sum = torch.sum(exp_x, dim=dim, keepdim=True)
    return exp_x / exp_sum

def log_softmax(x: Tensor, dim: int) -> Tensor:
    x = x - torch.max(x, dim=dim, keepdim=True).values
    return x - torch.log(torch.sum(torch.exp(x), dim=dim, keepdim=True))

def cross_entropy_loss(inputs: Float[Tensor, "batch_size vocab_size"], targets: Int[Tensor, "batch_size"]) -> Float[Tensor, ""]:
    negative_log_softmax = -log_softmax(inputs, -1)
    return torch.mean(torch.gather(negative_log_softmax, dim=-1, index=targets.unsqueeze(-1)))

def clip_gradient(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float, epsilon: float = 1e-6) -> None:
    l2_norm = torch.tensor(0.0, device=parameters[0].device)
    grads = []
    for p in parameters:
        if p.grad is not None:
            l2_norm += torch.sum(p.grad.data ** 2)
            grads.append(p.grad)
    
    l2_norm = torch.sqrt(l2_norm)
    coefficient = min(1, max_l2_norm / (l2_norm + epsilon))
    for grad in grads:
        grad.data *= coefficient