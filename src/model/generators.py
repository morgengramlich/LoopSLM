import torch
from torch import nn
from .basis import DeltaSurface3D, DeltaSurface4D
from .utils import nyquist_check_depth


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

    def generate_harmonics(self):
        R, C, D, amp = self.weight_diff_gen(self.row_coords, self.col_coords, self.depth_coords)
        raw_depth = D * amp.unsqueeze(1) * self.weight_depth_scale
        cum_depth = torch.zeros_like(raw_depth)
        cum_depth[:, 1:] = torch.cumsum(raw_depth[:, 1:], dim=1)
        return R, C, cum_depth


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

    def generate_harmonics(self):
        S, R, C, D, amp = self.diff_gen(self.mat_coords, self.row_coords, self.col_coords, self.depth_coords)
        raw_depth = D * amp.unsqueeze(1) * self.depth_scale
        cum_depth = torch.zeros_like(raw_depth)
        cum_depth[:, 1:] = torch.cumsum(raw_depth[:, 1:], dim=1)
        return S, R, C, cum_depth


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

    def generate_harmonics(self):
        R_dn, C_dn, S_dn, cum_dn = self._generate_harmonics_per_component(
            self.down_sel_coords, self.latent_coords, self.model_coords, self.depth_coords, self.down_gen
        )
        R_up, C_up, S_up, cum_up = self._generate_harmonics_per_component(
            self.up_sel_coords, self.model_coords, self.latent_coords, self.depth_coords, self.up_gen
        )

        R_out, C_out, D_out, amp_out = self.out_gen(self.model_coords, self.model_coords, self.depth_coords)
        raw_out = D_out * amp_out.unsqueeze(1) * self.depth_scale
        cum_out = torch.zeros_like(raw_out)
        cum_out[:, 1:] = torch.cumsum(raw_out[:, 1:], dim=1)

        return (R_dn, C_dn, S_dn, cum_dn), (R_up, C_up, S_up, cum_up), (R_out, C_out, cum_out)

    def _generate_harmonics_per_component(self, coord1, coord2, coord3, dept_coord, gen_fn):
        S, R, C, D, amp = gen_fn(coord1, coord2, coord3, dept_coord)
        raw_depth = D * amp.unsqueeze(1) * self.depth_scale
        cum_depth = torch.zeros_like(raw_depth)
        cum_depth[:, 1:] = torch.cumsum(raw_depth[:, 1:], dim=1)
        return (R, C, S, cum_depth)
