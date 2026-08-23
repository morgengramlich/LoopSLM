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
