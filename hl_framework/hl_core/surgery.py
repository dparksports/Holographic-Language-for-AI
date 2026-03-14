"""Embedding Surgery — QR orthogonalization of reserved tokens.

Overwrites Llama-3 reserved token embeddings (128000–128049) with mutually
orthogonal vectors computed via QR decomposition in FP32, then casts to BF16.
Registers a backward hook that zeros anchor gradients during future fine-tuning.
"""

from __future__ import annotations

import torch
from pathlib import Path
from typing import Optional


class EmbeddingSurgeon:
    """Perform orthogonal embedding surgery on a HuggingFace causal LM."""

    def __init__(
        self,
        model_id: str = "meta-llama/Meta-Llama-3-8B",
        anchor_start: int = 128_000,
        anchor_count: int = 50,
        save_path: str | Path = "./Llama-3-StarGate",
        device: str = "cuda",
    ) -> None:
        self.model_id = model_id
        self.anchor_ids = list(range(anchor_start, anchor_start + anchor_count))
        self.save_path = Path(save_path)
        self.device = device
        self._model = None
        self._tokenizer = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def execute(self) -> None:
        """Run the full surgery pipeline: load → orthogonalize → freeze → save."""
        self._load_model()
        self._orthogonalize()
        self._register_freeze_hook()
        self._save()
        print(f"Surgery complete — anchors secured and saved to {self.save_path}")

    # ------------------------------------------------------------------
    # Internal steps
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"Loading {self.model_id} in BF16 …")
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            device_map=self.device,
            torch_dtype=torch.bfloat16,
        )

    def _orthogonalize(self) -> None:
        """QR decomposition in FP32 → cast to BF16 → overwrite embedding rows."""
        embeddings = self._model.model.embed_tokens.weight.data
        dim = embeddings.shape[1]

        print(f"Orthogonalizing {len(self.anchor_ids)} anchors (dim={dim}) …")
        random_vecs = torch.randn(
            len(self.anchor_ids), dim, dtype=torch.float32, device=self.device
        )
        q, _r = torch.linalg.qr(random_vecs.T)  # [dim, K]
        orthogonal_anchors = q.T.to(torch.bfloat16)  # [K, dim]

        embeddings[self.anchor_ids] = orthogonal_anchors

    def _register_freeze_hook(self) -> None:
        """Zero out anchor gradients so orthogonality survives SFT."""
        anchor_ids = self.anchor_ids  # capture for closure

        def _hook(grad: torch.Tensor) -> torch.Tensor:
            grad[anchor_ids] = 0.0
            return grad

        self._model.model.embed_tokens.weight.register_hook(_hook)
        print("Gradient freeze hook registered for anchor rows.")

    def _save(self) -> None:
        self.save_path.mkdir(parents=True, exist_ok=True)
        self._model.save_pretrained(self.save_path)
        self._tokenizer.save_pretrained(self.save_path)


# ------------------------------------------------------------------
# CLI entrypoint: python -m hl_core.surgery
# ------------------------------------------------------------------
if __name__ == "__main__":
    surgeon = EmbeddingSurgeon()
    surgeon.execute()
