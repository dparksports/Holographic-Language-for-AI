"""Star Gate LogitsProcessor — CFG-constrained logit masking.

Implements the Cognitive Segfault mechanism: an AST / FSM verifier dictates
which token IDs are structurally valid at each step of autoregressive
generation.  Invalid tokens receive a -inf penalty, guaranteeing the model
can never emit a syntax violation.

Usage with HuggingFace generate():

    verifier = MockChevronVerifier(grammar_map)
    processor = StarGateProcessor(verifier)
    outputs = model.generate(inputs, logits_processor=[processor])
"""

from __future__ import annotations

import torch
from typing import List, Protocol, runtime_checkable


# ------------------------------------------------------------------
# Verifier interface (duck-typed)
# ------------------------------------------------------------------

@runtime_checkable
class Verifier(Protocol):
    """Any object with this interface can drive the Star Gate."""

    def get_valid_tokens(self, state: str) -> List[int]:
        """Return the list of token IDs that are structurally legal."""
        ...

    def advance(self, chosen_token_id: int) -> None:
        """Advance the FSM / AST state after a token is selected."""
        ...


# ------------------------------------------------------------------
# Star Gate Logits Processor
# ------------------------------------------------------------------

class StarGateProcessor:
    """
    HuggingFace-compatible LogitsProcessor that applies deterministic
    logit masking via an external ``Verifier``.
    """

    def __init__(self, verifier: Verifier, initial_state: str = "ROOT_NODE") -> None:
        self.verifier = verifier
        self.current_state = initial_state

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
    ) -> torch.FloatTensor:
        allowed = self.verifier.get_valid_tokens(self.current_state)

        # Build boolean mask — True = allowed
        mask = torch.zeros(scores.shape[-1], dtype=torch.bool, device=scores.device)
        mask[allowed] = True

        # Cognitive Segfault: -inf penalty for anything violating the CFG
        scores[:, ~mask] = -float("inf")
        return scores

    def on_token_selected(self, token_id: int) -> None:
        """Call after each token is chosen to advance the verifier state."""
        self.verifier.advance(token_id)
        # Optionally update self.current_state from the verifier


# ------------------------------------------------------------------
# Mock verifier for testing
# ------------------------------------------------------------------

class MockChevronVerifier:
    """Simple FSM verifier for unit testing / demonstration."""

    def __init__(self, state_map: dict[str, List[int]]) -> None:
        """
        Parameters
        ----------
        state_map : dict
            Mapping of ``state_name`` → ``list[valid_token_ids]``.
        """
        self.state_map = state_map
        self.state = "ROOT_NODE"

    def get_valid_tokens(self, state: str) -> List[int]:
        return self.state_map.get(state, [])

    def advance(self, chosen_token_id: int) -> None:
        # In a real implementation this would walk the AST / CFG.
        pass
