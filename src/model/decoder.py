from torch import nn
import torch.nn.functional as F
from .attention import MultiHeadAttention


class DynamicFFN(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, W_up, W_down, b_up=None, b_down=None):
        x = F.linear(x, W_up, b_up)
        x = F.silu(x)
        x = F.linear(x, W_down, b_down)
        return x


class DecoderBlock(nn.Module):
    """Dynamic Decoder block"""
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.norm1 = nn.RMSNorm(d_model)
        self.ffn = DynamicFFN()
        self.norm_ff = nn.RMSNorm(d_model)

    def forward(self, x, position_ids, W_q, W_k, W_v, W_o, W_up, W_down, b_up=None, b_down=None, attn_mask=None):
        x = x + self.dropout(
            self.self_attn(
                self.norm1(x), position_ids, W_q, W_k, W_v, W_o, is_causal=True, attn_mask=attn_mask
            )
        )
        x = x + self.dropout(self.ffn(self.norm_ff(x), W_up, W_down, b_up, b_down))
        return x
