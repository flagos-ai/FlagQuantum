"""Test-only interpreter for sparse states and balanced measurement branches."""

from cmath import exp
from collections import defaultdict
from math import sqrt, tau
from random import Random

import pytest

from flagquantum.core.classical_control import instruction_condition_clauses
from flagquantum.core.ir import CircuitIR


def execute_sparse_register(
    ir: CircuitIR, state: dict[int, complex], *, seed: int
) -> dict[int, complex]:
    """Execute the emitted gate set directly on occupied computational states."""
    rng = Random(seed)
    outcomes: dict[int, int] = {}
    for op in ir.instructions:
        clauses = instruction_condition_clauses(op)
        if clauses and not any(
            all(outcomes[bit] == expected for bit, expected in clause)
            for clause in clauses
        ):
            continue
        masks = tuple(1 << (ir.n_wires - q - 1) for q in op.wires)
        name = op.name
        if name == "h":
            (mask,) = masks
            updated: defaultdict[int, complex] = defaultdict(complex)
            for basis, value in state.items():
                updated[basis & ~mask] += value / sqrt(2)
                updated[basis | mask] += (-value if basis & mask else value) / sqrt(2)
            state = {
                basis: value for basis, value in updated.items() if abs(value) > 1e-16
            }
        elif name == "measure":
            (mask,) = masks
            outcome = rng.randrange(2)
            outcomes[op.metadata["classical_bit"]] = outcome
            state = {
                basis: value
                for basis, value in state.items()
                if bool(basis & mask) == outcome
            }
            probability = sum(abs(value) ** 2 for value in state.values())
            assert probability == pytest.approx(0.5, abs=1e-12)
            state = {basis: value / sqrt(probability) for basis, value in state.items()}
        elif name == "reset":
            (mask,) = masks
            state = {basis & ~mask: value for basis, value in state.items()}
        elif name in ("phase", "s", "sdg", "z", "cz", "t", "tdg"):
            mask = sum(masks)
            phases = {
                "s": 1j,
                "sdg": -1j,
                "z": -1,
                "cz": -1,
                "t": exp(1j * tau / 8),
                "tdg": exp(-1j * tau / 8),
            }
            phase = phases[name] if name in phases else exp(1j * op.params["theta"])
            state = {
                basis: phase * value if basis & mask == mask else value
                for basis, value in state.items()
            }
        elif name in ("x", "cx", "ccx"):
            control, target = sum(masks[:-1]), masks[-1]
            state = {
                basis ^ target if basis & control == control else basis: value
                for basis, value in state.items()
            }
        elif name == "cswap":
            control, left, right = masks
            state = {
                (
                    basis ^ (left | right)
                    if basis & control and bool(basis & left) != bool(basis & right)
                    else basis
                ): value
                for basis, value in state.items()
            }
        else:
            raise AssertionError(f"Unexpected gate {name}")
    return state
