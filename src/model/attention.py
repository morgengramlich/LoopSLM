from torch import nn
import torch.nn.functional as F
from .rope import RotaryPositionalEmbeddings


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention Layer using dynamically generated weights."""
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0, 'd_model must be divisible by n_heads'
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.rope = RotaryPositionalEmbeddings(self.d_k)

    def forward(self, x, position_ids, W_q, W_k, W_v, W_o, is_causal=False, attn_mask=None):
        B, S_q, E = x.shape

        q = F.linear(x, W_q)
        k = F.linear(x, W_k)
        v = F.linear(x, W_v)

        q = q.view(B, S_q, self.n_heads, self.d_k).transpose(1, 2)
        k = k.view(B, S_q, self.n_heads, self.d_k).transpose(1, 2)
        v = v.view(B, S_q, self.n_heads, self.d_k).transpose(1, 2)

        q, k = self.rope(q, k, position_ids)

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            is_causal=is_causal and attn_mask is None,
            dropout_p=0.0,
        )

        out = out.transpose(1, 2).contiguous().view(B, S_q, E)
        return F.linear(out, W_o)
