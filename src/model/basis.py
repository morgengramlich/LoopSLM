import math
import torch
from torch import nn

class DeltaCurve1D(nn.Module):
    def __init__(self,expansion_order, max_cycles = 5):
        super().__init__()
        self.expansion_order = expansion_order

        self.freqs = nn.Parameter(torch.empty(expansion_order))
        self.phases = nn.Parameter(torch.empty(expansion_order))

        max_freq = max_cycles * 2.0 * math.pi
        nn.init.uniform_(self.freqs, -max_freq, max_freq)
        nn.init.uniform_(self.phases, -math.pi, math.pi)

    def forward(self, coords):
        return torch.sin(self.freqs[:, None] * coords[None, :] + self.phases[:, None])


class DeltaSurface3D(nn.Module):
    def __init__(self, expansion_order, max_rc_cycles=5, max_depth_cycles=1, fan_in=None):
        super().__init__()
        self.row_freqs = nn.Parameter(torch.empty(expansion_order))
        self.col_freqs = nn.Parameter(torch.empty(expansion_order))
        self.depth_freqs = nn.Parameter(torch.empty(expansion_order))
        self.row_phases = nn.Parameter(torch.empty(expansion_order))
        self.col_phases = nn.Parameter(torch.empty(expansion_order))
        self.depth_phases = nn.Parameter(torch.empty(expansion_order))
        self.amplitudes = nn.Parameter(torch.empty(expansion_order))

        rc_max = max_rc_cycles * 2 * math.pi
        depth_max = max_depth_cycles * 2 * math.pi
        nn.init.uniform_(self.row_freqs, -rc_max, rc_max)
        nn.init.uniform_(self.col_freqs, -rc_max, rc_max)
        nn.init.uniform_(self.depth_freqs, -depth_max, depth_max)
        nn.init.uniform_(self.row_phases, -math.pi, math.pi)
        nn.init.uniform_(self.col_phases, -math.pi, math.pi)
        nn.init.uniform_(self.depth_phases, -math.pi, math.pi)

        E = expansion_order
        amp_std = (1.0 / math.sqrt(E)) if fan_in is None else (2.0 ** 1.5) / math.sqrt(E * fan_in)
        nn.init.normal_(self.amplitudes, mean=0.0, std=amp_std)

    def forward(self, row_coords, col_coords, depth_coords):
        row_terms = torch.sin(self.row_freqs.unsqueeze(1) * row_coords.unsqueeze(0) + self.row_phases.unsqueeze(1))
        col_terms = torch.sin(self.col_freqs.unsqueeze(1) * col_coords.unsqueeze(0) + self.col_phases.unsqueeze(1))
        depth_terms = torch.sin(self.depth_freqs.unsqueeze(1) * depth_coords.unsqueeze(0) + self.depth_phases.unsqueeze(1))
        return torch.einsum('e,er,ec,ed->rcd', self.amplitudes, row_terms, col_terms, depth_terms)


class DeltaSurface4D(nn.Module):
    def __init__(self, expansion_order, max_sel_cycles=1, max_rc_cycles=5,
                 max_depth_cycles=1, fan_in=None):
        super().__init__()
        self.sel_freqs = nn.Parameter(torch.empty(expansion_order))
        self.row_freqs = nn.Parameter(torch.empty(expansion_order))
        self.col_freqs = nn.Parameter(torch.empty(expansion_order))
        self.depth_freqs = nn.Parameter(torch.empty(expansion_order))
        self.sel_phases = nn.Parameter(torch.empty(expansion_order))
        self.row_phases = nn.Parameter(torch.empty(expansion_order))
        self.col_phases = nn.Parameter(torch.empty(expansion_order))
        self.depth_phases = nn.Parameter(torch.empty(expansion_order))
        self.amplitudes = nn.Parameter(torch.empty(expansion_order))

        sel_max = max_sel_cycles * 2 * math.pi
        rc_max = max_rc_cycles * 2 * math.pi
        depth_max = max_depth_cycles * 2 * math.pi
        nn.init.uniform_(self.sel_freqs, -sel_max, sel_max)
        nn.init.uniform_(self.row_freqs, -rc_max, rc_max)
        nn.init.uniform_(self.col_freqs, -rc_max, rc_max)
        nn.init.uniform_(self.depth_freqs, -depth_max, depth_max)
        nn.init.uniform_(self.sel_phases, -math.pi, math.pi)
        nn.init.uniform_(self.row_phases, -math.pi, math.pi)
        nn.init.uniform_(self.col_phases, -math.pi, math.pi)
        nn.init.uniform_(self.depth_phases, -math.pi, math.pi)

        E = expansion_order
        amp_std = (1.0 / math.sqrt(E)) if fan_in is None else (2.0 ** 2.0) / math.sqrt(E * fan_in)
        nn.init.normal_(self.amplitudes, mean=0.0, std=amp_std)

    def forward(self, sel_coords, row_coords, col_coords, depth_coords):
        sel_terms = torch.sin(self.sel_freqs.unsqueeze(1) * sel_coords.unsqueeze(0) + self.sel_phases.unsqueeze(1))
        row_terms = torch.sin(self.row_freqs.unsqueeze(1) * row_coords.unsqueeze(0) + self.row_phases.unsqueeze(1))
        col_terms = torch.sin(self.col_freqs.unsqueeze(1) * col_coords.unsqueeze(0) + self.col_phases.unsqueeze(1))
        depth_terms = torch.sin(self.depth_freqs.unsqueeze(1) * depth_coords.unsqueeze(0) + self.depth_phases.unsqueeze(1))
        return torch.einsum('e,es,er,ec,ed->srcd', self.amplitudes, sel_terms, row_terms, col_terms, depth_terms)
