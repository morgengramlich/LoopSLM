# LoopSLM

The repository contains a set of modules for building and experimenting with **LoopSLM** - a Small Language Model (SLM) architecture aiming to minimize training VRAM footprint and enable local execution.

The **LoopSLM** architecture consists of two core parts: a single base decoder-only layer and a set of hypersurfaces used to generate the diffs ($\Delta W_l$) that are dynamically applied to the base weight matrices to produce the next layer's weights ($W_l = W_0 + \Delta W_l$).
