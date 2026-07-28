from torch import nn
from attention import MultiHeadAttention

class DecoderBlock(nn.Module):
    """Decoder block"""
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.norm1 = nn.RMSNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.SiLU(),
            nn.Linear(d_ff, d_model),
        )
        self.norm_ff = nn.RMSNorm(d_model)

    def forward(self, x, position_ids, attn_mask=None):
        x = x + self.dropout(
            self.self_attn(
                self.norm1(x), position_ids, is_causal=True, attn_mask=attn_mask
            )
        )
        x = x + self.dropout(self.ffn(self.norm_ff(x)))
        return x
