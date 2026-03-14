"""Experiment A — Participation Ratio: polysemous English vs. orthogonal anchors.

Demonstrates Adversarial Polysemy by comparing the Participation Ratio (PR)
of 1,000 standard English token embeddings against 50 QR-orthogonalized
anchor vectors.  Higher PR = more diffuse / overlapping energy; lower PR =
sharper point-mass concentration.

Run:  python -m experiments.exp_a_polysemy   (from hl_framework/)
"""

from __future__ import annotations

import torch


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def calc_participation_ratio(covariance: torch.Tensor) -> float:
    """PR = (Σ λ_i)² / Σ λ_i²   over the eigenvalues of *covariance*."""
    eigenvalues = torch.linalg.eigvalsh(covariance)
    eigenvalues = torch.relu(eigenvalues)  # clean negative numerical dust
    return (eigenvalues.sum() ** 2 / (eigenvalues ** 2).sum()).item()


def generate_1k_english_token_ids() -> list[int]:
    """Extract token IDs for ~1,000 pure-alpha English subwords from the tokenizer.

    Falls back to the range [1000, 2000] when the tokenizer is unavailable.
    """
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
        vocab = tokenizer.get_vocab()
        sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])

        ids: list[int] = []
        for word, token_id in sorted_vocab:
            clean = word.replace("\u0120", "").strip()
            if clean.isalpha() and clean.islower() and len(clean) > 3:
                ids.append(token_id)
            if len(ids) == 1000:
                break
        return ids
    except Exception:
        # Fallback when model weights / auth are unavailable
        return list(range(1000, 2000))


# ------------------------------------------------------------------
# Main experiment
# ------------------------------------------------------------------

def run() -> None:
    print("Loading Llama-3-8B embeddings …")

    try:
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Meta-Llama-3-8B", torch_dtype=torch.float32
        )
        embeddings = model.model.embed_tokens.weight.detach()
    except Exception:
        print("  (Model unavailable — using random surrogate embeddings)")
        embeddings = torch.randn(128256, 4096)

    # 1. Standard English tokens
    eng_ids = generate_1k_english_token_ids()
    english_tokens = embeddings[eng_ids]

    # 2. Perfectly orthogonal anchors via QR
    dim = embeddings.shape[1]
    q, _r = torch.linalg.qr(torch.randn(dim, 50))
    orthogonal_anchors = q.T  # [50, dim]

    # Covariance → PR
    pr_eng = calc_participation_ratio(torch.cov(english_tokens.T))
    pr_anc = calc_participation_ratio(torch.cov(orthogonal_anchors.T))

    print()
    print("═" * 56)
    print("  Experiment A — Participation Ratio Comparison")
    print("═" * 56)
    print(f"  Standard English Tokens PR: {pr_eng:>8.2f}  (diffuse / overlapping)")
    print(f"  Orthogonal Anchors PR:      {pr_anc:>8.2f}  (point-mass)")
    print("═" * 56)


if __name__ == "__main__":
    run()
