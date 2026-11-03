"""High-level constrained synthesis, with Services as its implementation owner.

The initial unreleased profile supports coherent unsigned arithmetic on exact
CPU statevectors. See README.md for scope and the independent user-trial gate.
"""

from ..services.synthesis import (
    CRX,
    CX,
    SWAP,
    Const,
    ConstructionBudget,
    InputState,
    Output,
    Program,
    QBit,
    QNum,
    SynthesisError,
    UInt,
    adjoint,
    allocate,
    apply,
    hadamard_transform,
    phase,
    qfunc,
    synthesize,
    within,
)

__all__ = (
    "ConstructionBudget",
    "InputState",
    "Program",
    "SynthesisError",
    "UInt",
    "qfunc",
    "synthesize",
    "Output",
    "Const",
    "QNum",
    "QBit",
    "allocate",
    "hadamard_transform",
    "phase",
    "within",
    "apply",
    "adjoint",
    "CX",
    "SWAP",
    "CRX",
)
