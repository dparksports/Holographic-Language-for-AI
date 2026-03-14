"""Triton SRAM fused L2 Hopfield cleanup kernel.

Implements a continuous Hopfield network that snaps a drifting noisy vector
to the closest pristine anchor in a small dictionary — entirely inside GPU
SRAM / L1 cache.  The kernel performs a brute-force L2 nearest-neighbour
search over K anchors in a single thread-block, avoiding global memory
round-trips.

A pure-PyTorch CPU fallback is included for environments without Triton.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

# ------------------------------------------------------------------
# Triton kernel (imported lazily to avoid hard dependency)
# ------------------------------------------------------------------

_TRITON_AVAILABLE = False

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True

    @triton.jit
    def _fused_l2_hopfield_kernel(
        noisy_ptr,
        dict_ptr,
        out_ptr,
        D: tl.constexpr,
        K: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Single-block L2 nearest-neighbour snap entirely in SRAM."""
        offsets = tl.arange(0, BLOCK_SIZE)
        mask = offsets < D

        # Load the noisy vector once
        noisy_vec = tl.load(noisy_ptr + offsets, mask=mask)

        min_dist = float("inf")
        best_idx = 0

        # Brute-force search over K anchors (all in L1 cache)
        for k in range(K):
            anchor = tl.load(dict_ptr + k * D + offsets, mask=mask)
            diff = noisy_vec - anchor
            dist = tl.sum(diff * diff, axis=0)
            if dist < min_dist:
                min_dist = dist
                best_idx = k

        # Snap to closest pristine anchor
        best_anchor = tl.load(dict_ptr + best_idx * D + offsets, mask=mask)
        tl.store(out_ptr + offsets, best_anchor, mask=mask)

except ImportError:
    pass


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def hopfield_cleanup(noisy: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
    """Snap *noisy* to the closest vector in *anchors* using Triton L2.

    Parameters
    ----------
    noisy : Tensor [D]
        The drifting / corrupted vector.
    anchors : Tensor [K, D]
        Dictionary of pristine anchor vectors.

    Returns
    -------
    Tensor [D]
        The closest anchor (copied, not a view).
    """
    if _TRITON_AVAILABLE and noisy.is_cuda:
        return _hopfield_cleanup_triton(noisy, anchors)
    return hopfield_cleanup_cpu(noisy, anchors)


def _hopfield_cleanup_triton(noisy: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
    D = noisy.shape[-1]
    K = anchors.shape[0]
    out = torch.empty_like(noisy)

    # BLOCK_SIZE must be ≥ D and a power of 2
    block = 1
    while block < D:
        block <<= 1

    _fused_l2_hopfield_kernel[(1,)](noisy, anchors, out, D=D, K=K, BLOCK_SIZE=block)
    return out


# ------------------------------------------------------------------
# CPU fallback (pure PyTorch)
# ------------------------------------------------------------------

def hopfield_cleanup_cpu(noisy: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
    """PyTorch-only nearest-neighbour snap (works on CPU and CUDA)."""
    dists = torch.sum((anchors - noisy.unsqueeze(0)) ** 2, dim=-1)  # [K]
    best = dists.argmin()
    return anchors[best].clone()


# ------------------------------------------------------------------
# Continuous Hopfield cleanup (softmax-based, used by experiments)
# ------------------------------------------------------------------

def continuous_hopfield_cleanup(
    noisy: torch.Tensor,
    dictionary: torch.Tensor,
    beta: float = 15.0,
) -> torch.Tensor:
    """Softmax-attention Hopfield cleanup (Ramsauer et al., 2021).

    Parameters
    ----------
    noisy : Tensor [D]
        Noisy / drifting vector.
    dictionary : Tensor [K, D]
        Pristine anchor dictionary.
    beta : float
        Inverse temperature (higher = sharper attractor basins).

    Returns
    -------
    Tensor [D]
        Cleaned, L2-normalized vector.
    """
    similarities = F.linear(noisy, dictionary)  # [K]
    weights = F.softmax(beta * similarities, dim=-1)  # [K]
    cleaned = torch.matmul(weights, dictionary)  # [D]
    return F.normalize(cleaned, p=2, dim=-1)
