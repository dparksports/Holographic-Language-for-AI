"""Experiment C — 1,000-step drift comparison: raw FP32 vs Hopfield-stabilized.

Injects Gaussian noise into a vector for 1,000 steps and compares the cosine
similarity to the original anchor when:
  (a) accumulating naïvely (pure FP32 drift), versus
  (b) applying continuous Hopfield cleanup after each step.

Uses a pure-PyTorch softmax Hopfield network (no Triton dependency).

Run:  python -m experiments.exp_c_drift   (from hl_framework/)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from hl_core.hopfield_triton import continuous_hopfield_cleanup


def run(
    dim: int = 4096,
    num_anchors: int = 50,
    steps: int = 1000,
    noise_scale: float = 0.05,
    report_every: int = 200,
) -> None:
    # Pristine anchor dictionary
    dictionary = F.normalize(torch.randn(num_anchors, dim), p=2, dim=-1)
    target = dictionary[0].clone()

    drift_vec = target.clone()
    clean_vec = target.clone()

    print()
    print("═" * 64)
    print("  Experiment C — Drift Comparison (1,000 recursive steps)")
    print("═" * 64)
    print(f"  {'Step':<8} │ {'FP32 Drift (unprotected)':<28} │ {'Hopfield Stabilized':<20}")
    print("─" * 64)

    for step in range(1, steps + 1):
        noise = torch.randn(dim) * noise_scale

        # (a) Pure accumulation — drift
        drift_vec = F.normalize(drift_vec + noise, p=2, dim=-1)

        # (b) Accumulation + Hopfield cleanup
        clean_vec = F.normalize(clean_vec + noise, p=2, dim=-1)
        clean_vec = continuous_hopfield_cleanup(clean_vec, dictionary)

        if step % report_every == 0:
            sim_drift = F.cosine_similarity(drift_vec, target, dim=0).item()
            sim_clean = F.cosine_similarity(clean_vec, target, dim=0).item()
            print(f"  {step:<8} │ {sim_drift:<28.6f} │ {sim_clean:<20.6f}")

    print("═" * 64)


if __name__ == "__main__":
    run()
