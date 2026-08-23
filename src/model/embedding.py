from torch import nn
import torch.nn.functional as F

class FactorizedEmbedding(nn.Module):
    def __init__(self, vocab_size, d_emb, d_model):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_emb = d_emb
        self.d_model = d_model

        self.emb = nn.Embedding(vocab_size, d_emb)
        self.proj = nn.Linear(d_emb, d_model, bias=False)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.proj.weight, mean=0.0, std=0.02)

    def forward(self, idx):
        return self.proj(self.emb(idx))

    def compute_logits(self, x):
        h = F.linear(x, self.proj.weight.T)
        return F.linear(h, self.emb.weight)

    def get_full_weight(self):
        return self.emb.weight @ self.proj.weight.T
