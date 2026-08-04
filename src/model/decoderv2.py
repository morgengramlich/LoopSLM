from torch import nn
from .attention import LatentMultiHeadAttention
from .decoder import DynamicSwiGLU


class DecoderBlock(nn.Module):
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
