"""FHRR complex-phase algebra — O(d) bind / unbind in frequency space.

Frequency Holographic Reduced Representations (FHRR) encode information as
complex unit-circle vectors.  Binding is element-wise complex multiplication
(angle *addition*); unbinding is multiplication by the conjugate (angle
*subtraction*).  Both are O(d) — no FFT required.

This module is **hardware-agnostic** and runs on CPU or CUDA.
"""

from __future__ import annotations

import math
import torch
from typing import Optional


class FHRRAlgebra:
    """FHRR vector-symbolic algebra on complex unit-circle representations."""

    def __init__(
        self,
        dim: int = 4096,
        device: str = "cpu",
        dtype: torch.dtype = torch.complex64,
    ) -> None:
        self.dim = dim
        self.device = device
        self.dtype = dtype

    # ------------------------------------------------------------------
    # Primitives
    # ------------------------------------------------------------------

    def generate_anchor(self) -> torch.Tensor:
        """Create a random unitary vector on the complex unit circle.

        Each component has magnitude 1 and a uniformly random phase ∈ [-π, π).
        """
        phases = torch.rand(self.dim, device=self.device) * (2 * math.pi) - math.pi
        return torch.polar(
            torch.ones(self.dim, device=self.device),
            phases,
        ).to(self.dtype)

    def bind(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """O(d) binding — element-wise complex multiplication (angle addition)."""
        return a * b

    def unbind(self, bound: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        """O(d) unbinding — element-wise conjugate multiplication (angle subtraction)."""
        return bound * torch.conj(key)

    # ------------------------------------------------------------------
    # Interface helpers
    # ------------------------------------------------------------------

    def to_real_space(self, complex_vec: torch.Tensor) -> torch.Tensor:
        """Project a complex-d vector to a real-2d vector for MLP interfacing.

        A ``[4096]`` complex vector becomes an ``[8192]`` real vector via
        ``cat([real, imag])``.
        """
        return torch.cat([complex_vec.real, complex_vec.imag], dim=-1)

    @staticmethod
    def similarity(a: torch.Tensor, b: torch.Tensor) -> float:
        """Cosine similarity on the real projections of two complex vectors."""
        a_real = torch.cat([a.real, a.imag], dim=-1)
        b_real = torch.cat([b.real, b.imag], dim=-1)
        cos = torch.nn.functional.cosine_similarity(a_real, b_real, dim=-1)
        return cos.item() if cos.dim() == 0 else cos.mean().item()
