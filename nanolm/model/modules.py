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
    def __init__(self, d_model: int, num_heads: int, positional_embedding: RotaryPositionalEmbedding | None, device=None, dtype=None):
        super().__init__()
        self.w_q = Linear(d_model, d_model, device=device, dtype=dtype)
        self.w_k = Linear(d_model, d_model, device=device, dtype=dtype)
        self.w_v = Linear(d_model, d_model, device=device, dtype=dtype)
        self.w_o = Linear(d_model, d_model, device=device, dtype=dtype)
        self.num_heads = num_heads
        self.positional_embedding = positional_embedding
    
    def forward(self, x: Float[Tensor, "batch_size seq_len d_model"], token_positions: Int[Tensor, "batch_size seq_len"]) -> Float[Tensor, "batch_size seq_len d_model"]:
        q_proj = self.w_q(x)
        k_proj = self.w_k(x)
        v_proj = self.w_v(x)

        q_proj = rearrange(q_proj, 'b seq_len (h d_k) -> b h seq_len d_k', h=self.num_heads)
        k_proj = rearrange(k_proj, 'b seq_len (h d_k) -> b h seq_len d_k', h=self.num_heads)
        v_proj = rearrange(v_proj, 'b seq_len (h d_v) -> b h seq_len d_v', h=self.num_heads)

        # We must rearrange the projected Q and K before applying the positional embedding
        # Because for every head, the dimensio index is from 0 to d_k-1, which is the expected input dimension for the positional embedding.
        # If we apply the positional embedding before rearranging, the dimension index for the positional embedding will be from 0 to d_model-1, which is not what the positional embedding expects and will lead to incorrect positional encoding.
        if self.positional_embedding is not None:
            if token_positions is not None:
                # The token_positions tensor has the shape (batch_size, seq_len), and we need to reshape it to (batch_size, 1, seq_len) to match the expected input shape for the positional embedding. 
                # RoPE will raise an error if we don't do this.
                token_positions = rearrange(token_positions, '... seq_len -> ... 1 seq_len')
            q_proj = self.positional_embedding(q_proj, token_positions)
            k_proj = self.positional_embedding(k_proj, token_positions)

        casual_mask = torch.tril(torch.ones((x.shape[1], x.shape[1]), dtype=torch.bool, device=x.device))
        attn_output = scaled_dot_product_attention(q_proj, k_proj, v_proj, mask=casual_mask)
        attn_output = rearrange(attn_output, 'b h seq_len d_v -> b seq_len (h d_v)', h=self.num_heads)
        return self.w_o(attn_output)

class TransformerLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, positional_embedding: RotaryPositionalEmbedding | None, device=None, dtype=None):
        super().__init__()
        self.mha = MultiHeadSelfAttention(d_model, num_heads, positional_embedding, device, dtype)
        self.ffn = FFN(d_model, d_ff, device, dtype)
        self.mha_norm = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn_norm = RMSNorm(d_model, device=device, dtype=dtype)

    def forward(self, x: Float[Tensor, "batch_size seq_len d_model"], token_positions: Int[Tensor, "batch_size seq_len"]) -> Float[Tensor, "batch_size seq_len d_model"]:
        mha_input = self.mha_norm(x)
        mha_output = self.mha(mha_input, token_positions)
        norm_2_input = mha_output + x
        ffn_input = self.ffn_norm(norm_2_input)
        ffn_output = self.ffn(ffn_input)
        return norm_2_input + ffn_output
    
class TransformerLM(nn.Module):
    def __init__(
            self,
            vocab_size: int,
            context_length: int,
            num_layers: int,
            d_model: int,
            num_heads: int,
            d_ff: int,
            theta: float,
            device=None,
            dtype=None
    ):
        super().__init__()
        self.context_length = context_length
        self.device = device
        self.dtype = dtype
        d_k = d_model // num_heads
        self.embedding = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.rope = RotaryPositionalEmbedding(theta, d_k, context_length, device=device)

        self.layers = nn.ModuleList([
            TransformerLayer(d_model, num_heads, d_ff, self.rope, device=device, dtype=dtype) for _ in range(num_layers)
        ])

        self.final_norm = RMSNorm(d_model=d_model)
        self.lm_head = Linear(in_features=d_model, out_features=vocab_size)

    def forward(self, x: Int[Tensor, "batch_size seq_len"]) -> Float[Tensor, "batch_size seq_len d_model"]:
        # cannot use context length in forward function because the input sequence length can be different from the context length during inference,
        #  and we want to be able to handle variable-length input sequences without being constrained by a fixed context length. 
        # By using the actual sequence length of the input, we can ensure that the model can process sequences of varying lengths without running into issues related to exceeding a predefined context length.
        position_tokens = torch.arange(x.shape[1], device=self.device, dtype=self.dtype).unsqueeze(0).expand(x.shape[0], -1)
        x = self.embedding(x)

        for layer in self.layers:
            x = layer(x, position_tokens)
        x = self.final_norm(x)
        return self.lm_head(x)