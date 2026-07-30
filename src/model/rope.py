import torch
from torch import nn


class RotaryPositionalEmbeddings(nn.Module):
    """Rotary Positional Embedding (RoPE) — parameter-free, dynamic sequence length."""
    def __init__(self, d_k, max_seq_len=2048, base=10000):
        super().__init__()
        self.d_k = d_k
        self.base = base
        self._seq_len_cached = 0

        inv_freq = 1.0 / (base ** (torch.arange(0, d_k, 2).float() / d_k))
        self.register_buffer('inv_freq', inv_freq, persistent=False)

        self.register_buffer('cos_cache', torch.empty(0), persistent=False)
        self.register_buffer('sin_cache', torch.empty(0), persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len):
        if seq_len <= self._seq_len_cached:
            return

        positions = torch.arange(
            seq_len,
            device=self.inv_freq.device,
            dtype=self.inv_freq.dtype,
        )

        freqs = torch.outer(positions, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)

        self.cos_cache = emb.cos()
        self.sin_cache = emb.sin()
        self._seq_len_cached = seq_len

    def _rotate_half(self, x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, q, k, position_ids):
        max_seq_len = position_ids.max()
        if max_seq_len >= self._seq_len_cached:
            new_len = max(int(max_seq_len) + 1, self._seq_len_cached * 2)
            self._build_cache(new_len)

        cos = self.cos_cache[position_ids].unsqueeze(1).to(q.dtype)
        sin = self.sin_cache[position_ids].unsqueeze(1).to(q.dtype)

        q_rot = q * cos + self._rotate_half(q) * sin
        k_rot = k * cos + self._rotate_half(k) * sin
        return q_rot, k_rot
