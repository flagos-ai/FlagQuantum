"""High-level constrained synthesis, with Services as its implementation owner.

The initial unreleased profile supports coherent unsigned arithmetic on exact
CPU statevectors. See README.md for scope and the independent user-trial gate.
"""

from ..services.synthesis import (
    ConstructionBudget,
    InputState,
    Program,
    SynthesisError,
    UInt,
    qfunc,
    synthesize,
)

__all__ = (
    "ConstructionBudget",
    "InputState",
    "Program",
    "SynthesisError",
    "UInt",
    "qfunc",
    "synthesize",
)
