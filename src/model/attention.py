from torch import nn
import torch.nn.functional as F
from .utils import build_layer_weight


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

    def forward(self, x, base_attn, factors_attn, layer_idx, is_causal=False, attn_mask=None):
        B, S_q, E = x.shape
        W_q, W_k, W_v, W_o = base_attn
        S, R, C, cum_depth = factors_attn

        K_layer = cum_depth[:, layer_idx]

        W_q_n = build_layer_weight(W_q, K_layer * S[:, 0], R, C, layer_idx)
        W_k_n = build_layer_weight(W_k, K_layer * S[:, 1], R, C, layer_idx)
        W_v_n = build_layer_weight(W_v, K_layer * S[:, 2], R, C, layer_idx)
        W_o_n = build_layer_weight(W_o, K_layer * S[:, 3], R, C, layer_idx)

        q = F.linear(x, W_q_n).view(B, S_q, self.n_heads, self.d_k).transpose(1, 2)
        k = F.linear(x, W_k_n).view(B, S_q, self.n_heads, self.d_k).transpose(1, 2)
        v = F.linear(x, W_v_n).view(B, S_q, self.n_heads, self.d_k).transpose(1, 2)

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
        return F.linear(out, W_o_n)


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

    def forward(self, x, base_attn, factors_attn, layer_idx, is_causal=False, attn_mask=None):
        B, S_q, _ = x.shape

        W_dq, W_dkv, W_uq, W_uk, W_uv, W_o = base_attn
        factors_down, factors_up, factors_out = factors_attn
        R_dn, C_dn, S_dn, cum_dn = factors_down
        R_up, C_up, S_up, cum_up = factors_up
        R_out, C_out, cum_out = factors_out

        K_dn = cum_dn[:, layer_idx]
        K_up = cum_up[:, layer_idx]
        K_out = cum_out[:, layer_idx]

        W_dq_n = build_layer_weight(W_dq, K_dn * S_dn[:, 0], R_dn, C_dn, layer_idx)
        W_dkv_n = build_layer_weight(W_dkv, K_dn * S_dn[:, 1], R_dn, C_dn, layer_idx)

        W_uq_n = build_layer_weight(W_uq, K_up * S_up[:, 0], R_up, C_up, layer_idx)
        W_uk_n = build_layer_weight(W_uk, K_up * S_up[:, 1], R_up, C_up, layer_idx)
        W_uv_n = build_layer_weight(W_uv, K_up * S_up[:, 2], R_up, C_up, layer_idx)
        W_o_n = build_layer_weight(W_o, K_out, R_out, C_out, layer_idx)

        c_q = F.linear(x, W_dq_n)
        c_kv = F.linear(x, W_dkv_n)
        c_q = self.cq_norm(c_q)
        c_kv = self.ckv_norm(c_kv)

        q = F.linear(c_q, W_uq_n)
        k = F.linear(c_kv, W_uk_n)
        v = F.linear(c_kv, W_uv_n)

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
        return F.linear(out, W_o_n)
