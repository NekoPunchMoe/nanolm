import torch
from torch import Tensor
from jaxtyping import Float, Int

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
    