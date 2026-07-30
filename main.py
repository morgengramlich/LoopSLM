import torch
from transformers import AutoTokenizer
from src.model.delta import MODEL_V0

@torch.no_grad()
def reply(model, tokenizer, prompt, max_new_tokens=60, temperature=0.8, top_k=40, device=None):
    if device is None:
        device = next(model.parameters()).device

    idx = tokenizer.encode(prompt, add_special_tokens=False)
    idx = torch.tensor(idx[-model.max_seq_len:], dtype=torch.long, device=device).unsqueeze(0)
    prompt_len = idx.shape[1]

    out_ids = model.generate(idx, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k)

    new_ids = out_ids[0, prompt_len:].tolist()
    return tokenizer.decode(new_ids, skip_special_tokens=True)

def convert_checkpoint(old_checkpoint):
    sd = old_checkpoint['state_dict']
    new_sd = {}

    rename = {
        'llm_core.self_attn.W_q.weight': 'W_q',
        'llm_core.self_attn.W_k.weight': 'W_k',
        'llm_core.self_attn.W_v.weight': 'W_v',
        'llm_core.self_attn.W_o.weight': 'W_o',
        'llm_core.ffn.0.weight': 'W_up',
        'llm_core.ffn.2.weight': 'W_down',
        'llm_core.ffn.0.bias': 'b_up',
        'llm_core.ffn.2.bias': 'b_down',

        'llm_core.norm1.weight': 'decoder.norm1.weight',
        'llm_core.norm_ff.weight': 'decoder.norm_ff.weight',

        'norm_f.weight': 'norm_f.weight',
        'token_emb.weight': 'token_emb.weight',
    }

    for k, v in sd.items():
        if k in rename:
            new_sd[rename[k]] = v
        else:
            new_sd[k] = v

    old_checkpoint['state_dict'] = new_sd
    return old_checkpoint

if __name__ == '__main__':
    checkpoint = torch.load('./checkpoints/LoopLlmModel.pt', weights_only=False, map_location=torch.device('mps'))
    checkpoint = convert_checkpoint(checkpoint)

    model = MODEL_V0
    model = model.to(torch.device('mps'))
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    prompt = "Every effort moves you towards"
    response = reply(model, tokenizer, prompt, temperature=1, top_k=50, max_new_tokens=5, device=torch.device('mps'))
    print(prompt + response)
