import torch
from torch import nn
import torch.nn.functional as F

class RotaryPositionalEmbeddings(nn.Module):
    """Rotary Positional Embedding (RoPE) — parameter-free, dynamic sequence length."""
    def __init__(self, d_k, base=10000):
        super().__init__()
        self.d_k = d_k
        self.base = base

    def _get_freqs(self, position_ids):
        device = position_ids.device
        theta = 1.0 / (self.base ** (torch.arange(0, self.d_k, 2, device=device).float() / self.d_k))
        freqs = position_ids.unsqueeze(-1).float() * theta.view(1, 1, -1)
        freqs = torch.cat([freqs, freqs], dim=-1)
        return freqs

    def _rotate_half(self, x):
        half = x.shape[-1] // 2
        x1, x2 = x[..., :half], x[..., half:]
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, q, k, position_ids):
        freqs = self._get_freqs(position_ids)
        cos = freqs.cos().unsqueeze(1)
        sin = freqs.sin().unsqueeze(1)

        q_rot = q * cos + self._rotate_half(q) * sin
        k_rot = k * cos + self._rotate_half(k) * sin
        return q_rot, k_rot


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention Layer using dynamically generated weights."""
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0, 'd_model must be divisible by n_heads'
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        self.rope = RotaryPositionalEmbeddings(self.d_k)

    def forward(self, x, position_ids, context=None, is_causal=False, attn_mask=None):
        B, S_q, E = x.shape

        has_context = context is not None
        S_kv = context.shape[1] if has_context else S_q
        kv_input = context if has_context else x

        q = self.W_q(x)
        k = self.W_k(kv_input)
        v = self.W_v(kv_input)

        q = q.view(B, S_q, self.n_heads, self.d_k).transpose(1, 2)
        k = k.view(B, S_kv, self.n_heads, self.d_k).transpose(1, 2)
        v = v.view(B, S_kv, self.n_heads, self.d_k).transpose(1, 2)

        q, k = self.rope(q, k, position_ids)

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            is_causal=is_causal and attn_mask is None,
            dropout_p=0.0,
        )

        out = out.transpose(1, 2).reshape(B, S_q, E)
        return self.W_o(out)
