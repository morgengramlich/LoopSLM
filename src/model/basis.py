import math
import torch
from torch import nn


class DeltaSurface3D(nn.Module):
    def __init__(self, expansion_order, max_rc_cycles=5, max_depth_cycles=1):
        super().__init__()
        self.amplitudes = nn.Parameter(torch.zeros(expansion_order))
        self.row_freqs = nn.Parameter(torch.empty(expansion_order))
        self.col_freqs = nn.Parameter(torch.empty(expansion_order))
        self.depth_freqs = nn.Parameter(torch.empty(expansion_order))
        self.row_phases = nn.Parameter(torch.empty(expansion_order))
        self.col_phases = nn.Parameter(torch.empty(expansion_order))
        self.depth_phases = nn.Parameter(torch.empty(expansion_order))

        rc_max = max_rc_cycles * 2 * math.pi
        depth_max = max_depth_cycles * 2 * math.pi
        nn.init.uniform_(self.row_freqs, -rc_max, rc_max)
        nn.init.uniform_(self.col_freqs, -rc_max, rc_max)
        nn.init.uniform_(self.depth_freqs, -depth_max, depth_max)
        nn.init.uniform_(self.row_phases, -math.pi, math.pi)
        nn.init.uniform_(self.col_phases, -math.pi, math.pi)
        nn.init.uniform_(self.depth_phases, -math.pi, math.pi)

    def forward(self, row_coords, col_coords, depth_coords):
        row_terms = torch.sin(self.row_freqs.unsqueeze(1) * row_coords.unsqueeze(0) + self.row_phases.unsqueeze(1))
        col_terms = torch.sin(self.col_freqs.unsqueeze(1) * col_coords.unsqueeze(0) + self.col_phases.unsqueeze(1))
        depth_terms = torch.sin(self.depth_freqs.unsqueeze(1) * depth_coords.unsqueeze(0) + self.depth_phases.unsqueeze(1))
        return row_terms, col_terms, depth_terms, self.amplitudes


class DeltaSurface4D(nn.Module):
    def __init__(self, expansion_order, max_sel_cycles=1, max_rc_cycles=5, max_depth_cycles=1):
        super().__init__()
        self.amplitudes = nn.Parameter(torch.zeros(expansion_order))
        self.sel_freqs = nn.Parameter(torch.empty(expansion_order))
        self.row_freqs = nn.Parameter(torch.empty(expansion_order))
        self.col_freqs = nn.Parameter(torch.empty(expansion_order))
        self.depth_freqs = nn.Parameter(torch.empty(expansion_order))
        self.sel_phases = nn.Parameter(torch.empty(expansion_order))
        self.row_phases = nn.Parameter(torch.empty(expansion_order))
        self.col_phases = nn.Parameter(torch.empty(expansion_order))
        self.depth_phases = nn.Parameter(torch.empty(expansion_order))

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

    def forward(self, sel_coords, row_coords, col_coords, depth_coords):
        sel_terms = torch.sin(self.sel_freqs.unsqueeze(1) * sel_coords.unsqueeze(0) + self.sel_phases.unsqueeze(1))
        row_terms = torch.sin(self.row_freqs.unsqueeze(1) * row_coords.unsqueeze(0) + self.row_phases.unsqueeze(1))
        col_terms = torch.sin(self.col_freqs.unsqueeze(1) * col_coords.unsqueeze(0) + self.col_phases.unsqueeze(1))
        depth_terms = torch.sin(self.depth_freqs.unsqueeze(1) * depth_coords.unsqueeze(0) + self.depth_phases.unsqueeze(1))
        return sel_terms, row_terms, col_terms, depth_terms, self.amplitudes
