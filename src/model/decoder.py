from torch import nn
import torch.nn.functional as F
from .attention import MultiHeadAttention, LatentMultiHeadAttention
from .utils import build_layer_weight


class DynamicSwiGLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, W_up_base, W_down_base, factors_up, factors_down, layer_idx):
        R_up, C_up, cum_up = factors_up
        R_dn, C_dn, cum_dn = factors_down

        W_up_n = build_layer_weight(W_up_base, cum_up[:, layer_idx], R_up, C_up, layer_idx)
        W_dn_n = build_layer_weight(W_down_base, cum_dn[:, layer_idx], R_dn, C_dn, layer_idx)

        x = F.linear(x, W_up_n)
        gate, up = x.chunk(2, dim=-1)
        x = F.silu(gate) * up
        return F.linear(x, W_dn_n)


class DecoderBlock(nn.Module):
    """Dynamic Decoder block"""
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.norm1 = nn.RMSNorm(d_model)
        self.ffn = DynamicSwiGLU()
        self.norm_ff = nn.RMSNorm(d_model)

    def forward(self, x, base_attn, base_ffn, factors_attn, factors_up, factors_down, layer_idx, attn_mask=None):
        x = x + self.dropout(
            self.self_attn(
                self.norm1(x), base_attn, factors_attn, layer_idx, is_causal=True, attn_mask=attn_mask
            )
        )
        x = x + self.dropout(
            self.ffn(
                self.norm_ff(x), base_ffn[0], base_ffn[1], factors_up, factors_down, layer_idx
            )
        )
        return x

class LatentDecoderBlock(nn.Module):
    """Dynamic Decoder block"""
    def __init__(self, d_model, d_c, n_heads, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.self_attn = LatentMultiHeadAttention(d_model, d_c, n_heads)
        self.norm1 = nn.RMSNorm(d_model)
        self.ffn = DynamicSwiGLU()
        self.norm_ff = nn.RMSNorm(d_model)

    def forward(self, x, base_attn, base_ffn, factors_attn, factors_up, factors_down, layer_idx, attn_mask=None):
        x = x + self.dropout(
            self.self_attn(
                self.norm1(x), base_attn, factors_attn, layer_idx, is_causal=True, attn_mask=attn_mask
            )
        )

        W_up_base, W_down_base = base_ffn
        x = x + self.dropout(
            self.ffn(
                self.norm_ff(x), W_up_base, W_down_base, factors_up, factors_down, layer_idx
            )
        )
        return x
