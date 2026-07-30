import math
import torch
from torch import nn
import torch.nn.functional as F
from .generators import AttentionDeltaGenerator, LinearDeltaGenerator
from .decoder import DecoderBlock


class LoopLlm(nn.Module):
    def __init__(self, embeddings, n_heads, d_ff, num_layers, max_seq_len,
                 expansion_order, dropout=0.1):
        super().__init__()
        vocab_size, d_model = embeddings.shape
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len

        self.token_emb = nn.Embedding.from_pretrained(embeddings, freeze=True)
        self.dropout = nn.Dropout(dropout)

        self.W_q = nn.Parameter(torch.empty(d_model, d_model))
        self.W_k = nn.Parameter(torch.empty(d_model, d_model))
        self.W_v = nn.Parameter(torch.empty(d_model, d_model))
        self.W_o = nn.Parameter(torch.empty(d_model, d_model))
        self.W_up = nn.Parameter(torch.empty(d_ff, d_model))
        self.W_down = nn.Parameter(torch.empty(d_model, d_ff))
        self.b_up = nn.Parameter(torch.zeros(d_ff))
        self.b_down = nn.Parameter(torch.zeros(d_model))
        self._init_weights()

        self.attn_gen = AttentionDeltaGenerator(d_model, num_layers, expansion_order)
        self.ffn_up_gen = LinearDeltaGenerator(d_model, d_ff, num_layers, expansion_order)
        self.ffn_down_gen = LinearDeltaGenerator(d_ff, d_model, num_layers, expansion_order)

        self.decoder = DecoderBlock(d_model=d_model, n_heads=n_heads, dropout=dropout)
        self.norm_f = nn.RMSNorm(d_model)

    def _init_weights(self):
        """Initialize base transformer weights."""
        nn.init.xavier_uniform_(self.W_q)
        nn.init.xavier_uniform_(self.W_k)
        nn.init.xavier_uniform_(self.W_v)
        nn.init.xavier_uniform_(self.W_o)
        nn.init.xavier_uniform_(self.W_up)
        nn.init.xavier_uniform_(self.W_down)
        bound = 1 / math.sqrt(self.d_model)
        nn.init.uniform_(self.b_up, -bound, bound)
        bound = 1 / math.sqrt(self.d_ff)
        nn.init.uniform_(self.b_down, -bound, bound)

    def _base_weights(self):
        return (
            self.W_q, self.W_k, self.W_v, self.W_o,
            self.W_up, self.b_up,
            self.W_down, self.b_down
        )

    def forward(self, idx, position_ids, attn_mask=None):
        B, T = idx.shape
        assert T <= self.max_seq_len

        x = self.dropout(self.token_emb(idx))

        W_q, W_k, W_v, W_o, W_up, b_up, W_down, b_down = self._base_weights()
        for n in range(self.num_layers):
            if n > 0:
                d_qkvo = self.attn_gen.get_layer_diff(n)
                W_q = W_q + d_qkvo[0]
                W_k = W_k + d_qkvo[1]
                W_v = W_v + d_qkvo[2]
                W_o = W_o + d_qkvo[3]

                dW_up, db_up = self.ffn_up_gen.get_layer_diff(n)
                W_up = W_up + dW_up
                b_up = b_up + db_up if db_up is not None else b_up

                dW_down, db_down = self.ffn_down_gen.get_layer_diff(n)
                W_down = W_down + dW_down
                b_down = b_down + db_down if db_down is not None else b_down

            x = self.decoder(x, position_ids, W_q, W_k, W_v, W_o, W_up, W_down, b_up, b_down, attn_mask=attn_mask)

        x = self.norm_f(x)
        return x @ self.token_emb.weight.T

    def get_param_groups(self, base_lr, surface_lr):
        base_params = [self.W_q, self.W_k, self.W_v, self.W_o, self.W_up, self.b_up, self.W_down, self.b_down]
        base_params += list(self.decoder.parameters())
        base_params += list(self.norm_f.parameters())

        surface_params = (list(self.attn_gen.parameters()) +
                           list(self.ffn_up_gen.parameters()) +
                           list(self.ffn_down_gen.parameters()))

        base_ids = {id(p) for p in base_params}
        surface_ids = {id(p) for p in surface_params}
        accounted = base_ids | surface_ids
        for name, p in self.named_parameters():
            if p.requires_grad and id(p) not in accounted:
                raise ValueError(f"Unassigned parameter: {name}")

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
