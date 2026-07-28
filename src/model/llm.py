import torch
from torch import nn
import torch.nn.functional as F
from torch.func import functional_call

from generators import AttentionDeltaGenerator, LinearDeltaGenerator
from decoder import DecoderBlock


class LoopLlm(nn.Module):
    def __init__(self, embeddings, n_heads, d_ff, num_layers, max_seq_len,
                 expansion_order, dropout=0.1):
        super().__init__()
        vocab_size, d_model = embeddings.shape
        self.d_model = d_model
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.dropout = nn.Dropout(dropout)

        self.token_emb = nn.Embedding.from_pretrained(embeddings, freeze=True)

        self.attn_gen = AttentionDeltaGenerator(d_model, num_layers, expansion_order)
        self.ffn_up_gen = LinearDeltaGenerator(d_model, d_ff, num_layers, expansion_order)
        self.ffn_down_gen = LinearDeltaGenerator(d_ff, d_model, num_layers, expansion_order)

        self.llm_core = DecoderBlock(d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout=dropout)
        self.llm_core.norm1 = nn.RMSNorm(d_model)
        self.norm_f = nn.RMSNorm(d_model)

    def _base_weights(self):
        attn_layer = self.llm_core.self_attn
        base_q = attn_layer.W_q.weight
        base_k = attn_layer.W_k.weight
        base_v = attn_layer.W_v.weight
        base_o = attn_layer.W_o.weight
        base_qkvo = torch.stack([base_q, base_k, base_v, base_o], dim=0)

        base_up = self.llm_core.ffn[0].weight
        base_down = self.llm_core.ffn[2].weight
        return base_qkvo, base_up, base_down

    def _layer_params(self, running_qkvo, running_up, running_down):
        W_q, W_k, W_v, W_o = running_qkvo[0], running_qkvo[1], running_qkvo[2], running_qkvo[3]

        params = dict(self.llm_core.named_parameters())
        params['self_attn.W_q.weight'] = W_q
        params['self_attn.W_k.weight'] = W_k
        params['self_attn.W_v.weight'] = W_v
        params['self_attn.W_o.weight'] = W_o
        params['ffn.0.weight'] = running_up
        params['ffn.2.weight'] = running_down

        return params

    def forward(self, idx, position_ids, attn_mask=None):
        B, T = idx.shape
        assert T <= self.max_seq_len

        x = self.dropout(self.token_emb(idx))

        running_qkvo, running_up, running_down = self._base_weights()
        for n in range(self.num_layers):
            if n > 0:
                running_qkvo = running_qkvo + self.attn_gen.get_layer_diff(n)
                running_up = running_up + self.ffn_up_gen.get_layer_diff(n)
                running_down = running_down + self.ffn_down_gen.get_layer_diff(n)

            params = self._layer_params(running_qkvo, running_up, running_down)
            x = functional_call(
                self.llm_core, params, args=(x, position_ids),
                kwargs={'attn_mask': attn_mask}
            )

        x = self.norm_f(x)
        return x @ self.token_emb.weight.T

    def get_param_groups(self, base_lr, surface_lr):
        base_params = list(self.llm_core.parameters()) + list(self.norm_f.parameters())
        surface_params = (list(self.attn_gen.parameters()) +
                           list(self.ffn_up_gen.parameters()) +
                           list(self.ffn_down_gen.parameters()))

        base_ids = {id(p) for p in base_params}
        surface_ids = {id(p) for p in surface_params}
        accounted = base_ids | surface_ids
        for name, p in self.named_parameters():
            if p.requires_grad and id(p) not in accounted:
                raise ValueError(
                    f"'{name}' is trainable but not in either group -- "
                    f"get_param_groups is out of sync with the model's "
                    f"module structure and needs updating."
                )

        return [
            {'params': base_params, 'lr': base_lr, 'name': 'base'},
            {'params': surface_params, 'lr': surface_lr, 'name': 'surface'},
        ]

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Autoregressive generation loop for testing out the model."""
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.max_seq_len:]

            B, T = idx_cond.shape
            position_ids = torch.arange(0, T, dtype=torch.long, device=idx.device).unsqueeze(0).expand(B, T)

            logits = self(idx_cond, position_ids)[:, -1, :]
            logits = logits / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float("inf")

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
