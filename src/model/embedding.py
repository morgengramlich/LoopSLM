import math
import torch
from torch import nn
from basis import DeltaCurve1D

class HarmonicEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim, expansion_order, max_cycles=5):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.expansion_order = expansion_order

        self.basis = DeltaCurve1D(
            expansion_order=expansion_order,
            max_cycles=max_cycles,
        )

        self.coefficients = nn.Parameter(torch.empty(vocab_size, expansion_order))
        nn.init.normal_(self.coefficients, mean=0.0, std=1.0 / math.sqrt(expansion_order))
        self.register_buffer('coords', torch.linspace(-1.0, 1.0, embedding_dim))

    @property
    def weight(self):
        basis = self.basis(self.coords)
        return self.coefficients @ basis

    def forward(self, token_ids):
        basis = self.basis(self.coords)
        coeffs = self.coefficients[token_ids]
        return torch.einsum('bte,ed->btd', coeffs, basis)
