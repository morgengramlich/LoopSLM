from torch import nn
import torch.nn.functional as F
from .attention import MultiHeadAttention, LatentMultiHeadAttention


class DynamicSwiGLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, W_up_gate, W_down):
        x = F.linear(x, W_up_gate)
        gate, up = x.chunk(2, dim=-1)
        x = F.silu(gate) * up
        return F.linear(x, W_down)


class DecoderBlock(nn.Module):
    """Dynamic Decoder block"""
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.norm1 = nn.RMSNorm(d_model)
        self.ffn = DynamicSwiGLU()
        self.norm_ff = nn.RMSNorm(d_model)

    def forward(self, x, W_q, W_k, W_v, W_o, W_up, W_down, attn_mask=None):
        x = x + self.dropout(
            self.self_attn(
                self.norm1(x), W_q, W_k, W_v, W_o, is_causal=True, attn_mask=attn_mask
            )
        )
        x = x + self.dropout(self.ffn(self.norm_ff(x), W_up, W_down))
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

    def forward(self, x, W_dq, W_dkv, W_uq, W_uk, W_uv, W_o, W_up, W_down, attn_mask=None):
        x = x + self.dropout(
            self.self_attn(
                self.norm1(x), W_dq, W_dkv, W_uq, W_uk, W_uv, W_o, is_causal=True, attn_mask=attn_mask
            )
        )
        x = x + self.dropout(self.ffn(self.norm_ff(x), W_up, W_down))
        return x
