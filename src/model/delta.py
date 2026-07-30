from transformers import AutoModel
from .llm import LoopLlm

MODEL_V0 = LoopLlm(
    AutoModel.from_pretrained('gpt2').wte.weight.detach(),
    n_heads=12, d_ff=3072, num_layers=24, expansion_order=128,
    max_seq_len=1024
)

MODEL_V1 = LoopLlm(
    AutoModel.from_pretrained('gpt2').wte.weight.detach(),
    n_heads=24, d_ff=6144, num_layers=48, expansion_order=128,
    max_seq_len=2048
)
