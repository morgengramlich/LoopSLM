# LoopSLM

The repository contains a set of modules for building and experimenting with **LoopSLM** - a Small Language Model (SLM) architecture aiming to minimize training VRAM footprint and enable local execution.

The **LoopSLM** architecture consists of two core parts: a base decoder-only layer and a set of hypersurfaces used to generate the diffs ($\Delta W_l$) that are dynamically applied to the base weight matrices to produce the next layer's weights ($W_l = W_0 + \Delta W_l$).

# Model architecture

The model processes the input tensor by iteratively looping through its base layer $L$ times. At each iteration $l$, the base layer's weights are dynamically modified using diffs ($\Delta W_l$) obtained from the cross-section of a hypersurface.

## 1. The hypersurfaces

At a high level, the hypersurface is a continuous multi-dimensional function that generates weight differentials using a sum of spatial harmonic waves.Rather than storing explicit weight matrices, the surface landscape is constructed in three steps:

1. **1D Harmonic Waves:** For each dimension (row $r$, column $c$, layer depth $d$, and matrix selector $s$), the model generates 1D triangular waves ($\text{Tri}_i(dim)$) with learnable frequencies and phases.

2. **N-D Surface Components:** Multiplying the 1D wave values across all dimensions creates a single multi-dimensional surface component:

$$\text{Surface}_i(r, c, d) = \text{Tri}_i(r) \cdot \text{Tri}_i(c) \cdot \text{Tri}_i(d)$$

3. **Sum of Harmonics:** Stacking $E$ of these components (`expansion_order`) scaled by learned amplitudes ($A_i$) builds the complete hypersurface:

$$\Delta W(r, c, d) = \sum_{i=1}^{\text{expansion\_order}} A_i \cdot \left(\text{Surface}_i(r, c, d) \right)$$

By sampling this continuous space at fixed discrete coordinates $(r_l, c_l, d_l)$ for layer $l$, the model extracts a weight diff $\Delta W_l$ that modifies the shared base layer.

