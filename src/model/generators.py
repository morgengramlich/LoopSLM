import torch
from torch import nn
from .basis import DeltaSurface3D, DeltaSurface4D


def nyquist_check_depth(max_depth_cycles, num_layers, label=''):
    nyquist = num_layers / 2.0
    ratio = max_depth_cycles / nyquist
    flag = 'OK' if ratio < 0.5 else ('CAUTION' if ratio < 1.0 else 'LIKELY ALIASED')
    print(f"{label} max_depth_cycles={max_depth_cycles} vs depth-Nyquist={nyquist:.1f} "
          f"(num_layers={num_layers}) -> ratio {ratio:.2f} -> {flag}")


class LinearDeltaGenerator(nn.Module):
    def __init__(self, in_features, out_features, num_layers, expansion_order,
                 max_rc_cycles=5, max_depth_cycles=1):
        super().__init__()
        nyquist_check_depth(max_depth_cycles, num_layers, label='[ffn proj]')

        self.weight_diff_gen = DeltaSurface3D(
            expansion_order, max_rc_cycles=max_rc_cycles,
            max_depth_cycles=max_depth_cycles, fan_in=in_features
        )
        self.weight_depth_scale = nn.Parameter(torch.tensor(0.0))

        self.register_buffer('row_coords', torch.linspace(-1.0, 1.0, out_features))
        self.register_buffer('col_coords', torch.linspace(-1.0, 1.0, in_features))
        self.register_buffer('depth_coords', torch.linspace(-1.0, 1.0, num_layers))

    def get_layer_diff(self, layer_idx):
        d = self.depth_coords[layer_idx: layer_idx + 1]
        return self.weight_depth_scale * self.weight_diff_gen(self.row_coords, self.col_coords, d).squeeze(-1)


class AttentionDeltaGenerator(nn.Module):
    def __init__(self, d_model, num_layers, expansion_order,
                 max_mat_cycles=1, max_rc_cycles=5, max_depth_cycles=1):
        super().__init__()
        self.d_model = d_model
        nyquist_check_depth(max_depth_cycles, num_layers, label='[attention]')

        self.diff_gen = DeltaSurface4D(expansion_order, max_sel_cycles=max_mat_cycles,
                                  max_rc_cycles=max_rc_cycles, max_depth_cycles=max_depth_cycles,
                                  fan_in=d_model)
        self.depth_scale = nn.Parameter(torch.tensor(0.0))

        self.register_buffer('mat_coords', torch.linspace(-1.0, 1.0, 4))
        self.register_buffer('row_coords', torch.linspace(-1.0, 1.0, d_model))
        self.register_buffer('col_coords', torch.linspace(-1.0, 1.0, d_model))
        self.register_buffer('depth_coords', torch.linspace(-1.0, 1.0, num_layers))

    def get_layer_diff(self, layer_idx):
        d = self.depth_coords[layer_idx: layer_idx + 1]
        raw = self.diff_gen(self.mat_coords, self.row_coords, self.col_coords, d)
        return self.depth_scale * raw.squeeze(-1)


class LatentAttentionDeltaGenerator(nn.Module):
    def __init__(self, d_model, d_c, num_layers, expansion_order,
                 max_mat_cycles=1, max_rc_cycles=5, max_depth_cycles=1):
        super().__init__()
        self.d_model = d_model
        self.d_c = d_c

        nyquist_check_depth(max_depth_cycles, num_layers, label='[latent_attention]')

        self.down_gen = DeltaSurface4D(
            expansion_order, max_sel_cycles=max_mat_cycles,
            max_rc_cycles=max_rc_cycles, max_depth_cycles=max_depth_cycles,
            fan_in=d_model
        )
        self.up_gen = DeltaSurface4D(
            expansion_order, max_sel_cycles=max_mat_cycles,
            max_rc_cycles=max_rc_cycles, max_depth_cycles=max_depth_cycles,
            fan_in=d_c
        )
        self.out_gen = DeltaSurface3D(
            expansion_order, max_rc_cycles=max_rc_cycles,
            max_depth_cycles=max_depth_cycles, fan_in=d_model
        )
        self.depth_scale = nn.Parameter(torch.tensor(0.0))

        self.register_buffer('down_sel_coords', torch.linspace(-1.0, 1.0, 2))
        self.register_buffer('up_sel_coords', torch.linspace(-1.0, 1.0, 3))
        self.register_buffer('model_coords', torch.linspace(-1.0, 1.0, d_model))
        self.register_buffer('latent_coords', torch.linspace(-1.0, 1.0, d_c))
        self.register_buffer('depth_coords', torch.linspace(-1.0, 1.0, num_layers))

    def get_layer_diff(self, layer_idx):
        d = self.depth_coords[layer_idx: layer_idx + 1]

        raw_down = self.down_gen(self.down_sel_coords, self.latent_coords, self.model_coords, d).squeeze(-1)
        raw_up = self.up_gen(self.up_sel_coords, self.model_coords, self.latent_coords, d).squeeze(-1)
        raw_out = self.out_gen(self.model_coords, self.model_coords, d).squeeze(-1)

        s = self.depth_scale
        return (
            s * raw_down[0],
            s * raw_down[1],
            s * raw_up[0],
            s * raw_up[1],
            s * raw_up[2],
            s * raw_out,
        )
