"""hl_core — Holographic Language core primitives.

Modules:
    surgery         Embedding Surgery (QR orthogonalization + gradient freeze)
    stargate        Star Gate LogitsProcessor (CFG-constrained logit masking)
    fhrr_algebra    FHRR complex-phase algebra (bind / unbind / to_real_space)
    hopfield_triton Triton SRAM fused L2 Hopfield cleanup kernel
"""

__version__ = "0.1.0"
