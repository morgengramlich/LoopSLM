from torch import nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention Layer using dynamically generated weights."""
    def __init__(self, d_model, n_heads, qk_norm=True):
        super().__init__()
        assert d_model % n_heads == 0, 'd_model must be divisible by n_heads'
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        if qk_norm:
            self.q_norm = nn.RMSNorm(self.d_k, eps=1e-6, elementwise_affine=False)
            self.k_norm = nn.RMSNorm(self.d_k, eps=1e-6, elementwise_affine=False)
        else:
            self.q_norm = self.k_norm = None

    def forward(self, x, W_q, W_k, W_v, W_o, is_causal=False, attn_mask=None):
        B, S_q, E = x.shape

        q = F.linear(x, W_q)
        k = F.linear(x, W_k)
        v = F.linear(x, W_v)

        q = q.view(B, S_q, self.n_heads, self.d_k).transpose(1, 2)
        k = k.view(B, S_q, self.n_heads, self.d_k).transpose(1, 2)
        v = v.view(B, S_q, self.n_heads, self.d_k).transpose(1, 2)

        if self.q_norm:
            q = self.q_norm(q)
        if self.k_norm:
            k = self.k_norm(k)

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            is_causal=is_causal and attn_mask is None,
            dropout_p=0.0,
        )

        out = out.transpose(1, 2).contiguous().view(B, S_q, E)
        return F.linear(out, W_o)


class LatentMultiHeadAttention(nn.Module):
    """Multi-Head Latent Attention (MLA) using dynamically generated weights."""
    def __init__(self, d_model, d_c, n_heads):
        super().__init__()
        assert d_model % n_heads == 0, 'd_model must be divisible by n_heads'
        self.d_model = d_model
        self.d_c = d_c
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.cq_norm = nn.RMSNorm(d_c, eps=1e-6, elementwise_affine=False)
        self.ckv_norm = nn.RMSNorm(d_c, eps=1e-6, elementwise_affine=False)

    def forward(self, x, W_dq, W_dkv, W_uq, W_uk, W_uv, W_o, is_causal=False, attn_mask=None):
        B, S_q, _ = x.shape

        c_q = F.linear(x, W_dq)
        c_kv = F.linear(x, W_dkv)

        c_q = self.cq_norm(c_q)
        c_kv = self.ckv_norm(c_kv)

        q = F.linear(c_q, W_uq)
        k = F.linear(c_kv, W_uk)
        v = F.linear(c_kv, W_uv)

        q = q.view(B, S_q, self.n_heads, self.d_k).transpose(1, 2)
        k = k.view(B, S_q, self.n_heads, self.d_k).transpose(1, 2)
        v = v.view(B, S_q, self.n_heads, self.d_k).transpose(1, 2)

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            is_causal=is_causal and attn_mask is None,
            dropout_p=0.0,
        )

        out = out.transpose(1, 2).contiguous().view(B, S_q, self.d_model)
        return F.linear(out, W_o)
