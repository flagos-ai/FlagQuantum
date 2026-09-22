"""Shared deterministic programs for cross-framework semantic parity tests."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import torch

from ..circuit import Circuit
from ..core.ir import CircuitIR, Instruction

_SEEDS = (314, 2718, 5772, 8119)
_PARAMETERS = {"rx": "theta", "ry": "theta", "rz": "theta"}
_COMMON_OPERATIONS = (
    ("x", 1),
    ("y", 1),
    ("z", 1),
    ("h", 1),
    ("s", 1),
    ("t", 1),
    ("rx", 1),
    ("ry", 1),
    ("rz", 1),
    ("cx", 2),
    ("cz", 2),
    ("swap", 2),
)


@dataclass(frozen=True)
class SemanticParityCase:
    """One framework-neutral program shared by every optional SDK lane."""

    name: str
    program: CircuitIR


def _seeded_case(seed: int) -> SemanticParityCase:
    rng = random.Random(seed)
    n_wires = 3 + seed % 3
    instructions: list[Instruction] = [
        Instruction("ry", (wire,), params={"theta": rng.uniform(-math.pi, math.pi)})
        for wire in range(n_wires)
    ]
    for _ in range(16):
        name, arity = rng.choice(_COMMON_OPERATIONS)
        parameter = _PARAMETERS.get(name)
        instructions.append(
            Instruction(
                name,
                tuple(rng.sample(range(n_wires), arity)),
                params=(
                    {parameter: rng.uniform(-math.pi, math.pi)}
                    if parameter is not None
                    else {}
                ),
            )
        )
    return SemanticParityCase(
        f"shared_seed_{seed}",
        CircuitIR(
            n_wires,
            tuple(instructions),
            dtype="complex128",
            shape=(1, 2**n_wires),
        ),
    )


def semantic_parity_cases() -> tuple[SemanticParityCase, ...]:
    """Return the immutable corpus used by every supported framework lane."""

    asymmetric = SemanticParityCase(
        "shared_asymmetric_wire_order",
        CircuitIR(
            4,
            (
                Instruction("x", (3,)),
                Instruction("ry", (0,), params={"theta": 0.371}),
                Instruction("cx", (3, 1)),
                Instruction("rz", (2,), params={"theta": -0.219}),
                Instruction("swap", (0, 2)),
                Instruction("cz", (1, 0)),
            ),
            dtype="complex128",
            shape=(1, 16),
        ),
    )
    return (asymmetric, *(_seeded_case(seed) for seed in _SEEDS))


def reference_state(case: SemanticParityCase) -> torch.Tensor:
    """Evaluate one corpus case with FlagQuantum's complex128 simulator."""

    return Circuit.from_ir(case.program, dtype=torch.complex128).state()[0]


def maximum_error_up_to_global_phase(
    actual: torch.Tensor, expected: torch.Tensor
) -> float:
    """Return maximum amplitude error after removing one global phase."""

    actual = torch.as_tensor(actual, dtype=torch.complex128).reshape(-1)
    expected = torch.as_tensor(expected, dtype=torch.complex128).reshape(-1)
    if actual.shape != expected.shape:
        raise ValueError(
            f"statevector shapes differ: {tuple(actual.shape)} != {tuple(expected.shape)}"
        )
    anchor = int(torch.argmax(torch.abs(expected)).item())
    if torch.abs(expected[anchor]) == 0 or torch.abs(actual[anchor]) == 0:
        return float(torch.max(torch.abs(actual - expected)).item())
    phase = actual[anchor] / expected[anchor]
    phase = phase / torch.abs(phase)
    return float(torch.max(torch.abs(actual / phase - expected)).item())


__all__ = (
    "SemanticParityCase",
    "maximum_error_up_to_global_phase",
    "reference_state",
    "semantic_parity_cases",
)
