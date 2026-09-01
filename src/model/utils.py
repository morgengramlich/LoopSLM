import torch.nn.functional as F

def nyquist_check_depth(max_depth_cycles, num_layers, label=''):
    nyquist = num_layers / 2.0
    ratio = max_depth_cycles / nyquist
    flag = 'OK' if ratio < 0.5 else ('CAUTION' if ratio < 1.0 else 'LIKELY ALIASED')
    print(f"{label} max_depth_cycles={max_depth_cycles} vs depth-Nyquist={nyquist:.1f} "
          f"(num_layers={num_layers}) -> ratio {ratio:.2f} -> {flag}")


def build_layer_weight(base_weight, K, R, C, layer_idx):
    if layer_idx == 0:
        return base_weight
    dW = (R.T * K.unsqueeze(0)) @ C
    return base_weight + dW

def factorized_dynamic_linear(x, W_base, R, C, K_S, layer_idx, context_state=None):
    y_base = F.linear(x, W_base)
    if layer_idx == 0:
        return y_base

    z = F.linear(x, C)
    z_mod = z * K_S.view(1, 1, -1)
    if context_state is not None:
        z_mod = z_mod * (1.0 + context_state)
    y_delta = F.linear(z_mod, R.T)

    return y_base + y_delta
