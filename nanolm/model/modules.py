import math
import torch
from torch import nn, Tensor
from jaxtyping import Bool, Float, Int
from einops import einsum, rearrange
from utils import softmax

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__()
        std = math.sqrt(2.0 / (in_features + out_features))
        self.weight = nn.Parameter(nn.init.trunc_normal_(torch.empty((out_features, in_features), device=device, dtype=dtype), std=std, a=-3 * std, b=3 * std), requires_grad=True)

    def forward(self, x: Float[Tensor, "... d_in"]) -> Float[Tensor, "... d_out"]:
        return einsum(x, self.weight, '... d, h d -> ... h')
    
class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        super().__init__()
        std = math.sqrt(2.0 / (num_embeddings + embedding_dim))
        self.weight = nn.Parameter(nn.init.trunc_normal_(torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype), std=std, a=-3 * std, b=3 * std), requires_grad=True)

    def forward(self, token_ids: Int[Tensor, "..."]) -> Float[Tensor, "... d_model"]:
        return self.weight[token_ids]
    
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.gain = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype), requires_grad=True)
        self.eps = eps

    def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
        in_type = x.dtype
        x = x.to(torch.float32)
        # torch.rsqrt computes the reciprocal of the square root, which is more efficient than computing the square root and then taking the reciprocal separately.
        # The mean is computed along the last dimension (dim=-1) to get the mean squared value for each feature across the batch. The keepdim=True argument ensures that the output has the same number of dimensions as the input, which is necessary for broadcasting when multiplying with x.
        rms = torch.rsqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        x_normed = x * rms * self.gain
        return x_normed.to(in_type)
    
class FFN(nn.Module): # SwiGLU
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.linear1 = Linear(d_model, d_ff , device=device, dtype=dtype)
        self.linear_gate = Linear(d_model, d_ff , device=device, dtype=dtype)
        self.linear2 = Linear(d_ff, d_model, device=device, dtype=dtype)

    def silu(self, x: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
        return x * torch.sigmoid(x)
    
    def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
        x1 = self.silu(self.linear1(x))
        x_gate = self.linear_gate(x)
        x1 = x1 * x_gate
        return self.linear2(x1)

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device

        inv_freq = 1.0 / (theta ** (torch.arange(0, d_k, 2, device=device) / d_k))
        position = torch.arange(max_seq_len, device=device)
        theta_matrix = torch.einsum('i,j->ij', position, inv_freq)
        sin_pos = torch.sin(theta_matrix)
        cos_pos = torch.cos(theta_matrix)
        self.register_buffer('sin_pos', sin_pos, persistent=False)
        self.register_buffer('cos_pos', cos_pos, persistent=False)

    def forward(self, x: Float[Tensor, "... seq_len d_k"], token_positions: Int[Tensor, "... seq_len"]) -> Float[Tensor, "... seq_len d_k"]:
        sin_pos = self.sin_pos[token_positions, :]
        cos_pos = self.cos_pos[token_positions, :]
        x1, x2 = x[..., 0::2], x[..., 1::2]
        return torch.stack([x1 * cos_pos - x2 * sin_pos, x1 * sin_pos + x2 * cos_pos], dim=-1).flatten(-2, -1)
    
def scaled_dot_product_attention(
        Q: Float[Tensor, "batch_size ... query_len d_k"],
        K: Float[Tensor, "batch_size ... key_len d_k"],
        V: Float[Tensor, "batch_size ... key_len d_v"],
        mask: Bool[Tensor, "batch_size ... query_len key_len"] | None = None
    ) -> Float[Tensor, "batch_size ... query_len d_v"]:
    d_k = Q.shape[-1]
    attn_matrix = einsum(Q, K, "... query_len d_k, ... key_len d_k -> ... query_len key_len") / math.sqrt(d_k)
    
    if mask is not None:
        attn_matrix = torch.where(mask, attn_matrix, torch.tensor(float('-inf')))
    
    attn_weights = softmax(attn_matrix, dim=-1)
    return einsum(attn_weights, V, "... query_len key_len, ... key_len d_v -> ... query_len d_v")

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, device=None, dtype=None):
        super().__init__()
        self.w_q = Linear(d_model, d_model, device=device, dtype=dtype)
        self.w_k = Linear(d_model, d_model, device=device, dtype=dtype)
        self.w_v = Linear(d_model, d_model, device=device, dtype=dtype)
        self.w_o = Linear(d_model, d_model, device=device, dtype=dtype)
        self.num_heads = num_heads
