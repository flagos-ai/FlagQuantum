"""Rank-one planning proof admits closed basis permutations without guessing."""

from dataclasses import replace

import pytest

from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.runtime.planner.estimates import exact_basis_mps_bond

pytestmark = pytest.mark.unit


def test_basis_permutations_and_signs_have_rank_one() -> None:
    source = CircuitIR(
        n_wires=64,
        instructions=(
            Instruction("x", (0,)),
            Instruction("z", (0,)),
            Instruction("ccx", (0, 63, 2)),
        ),
    )
    assert exact_basis_mps_bond(source) == 1
    assert exact_basis_mps_bond(replace(source, instructions=())) == 1


@pytest.mark.parametrize(
    "gate",
    [
        Instruction("h", (0,)),
        Instruction("cx", (0, 63)),
        Instruction("swap", (0, 63)),
        Instruction("rx", (0,), params={"theta": 0.0}),
        Instruction("x", (0,), params={"theta": 0.0}),
        Instruction("x", (0,), metadata={"condition": 1}),
    ],
)
def test_unproven_gate_cannot_certify_rank_one(gate: Instruction) -> None:
    assert exact_basis_mps_bond(CircuitIR(n_wires=64, instructions=(gate,))) is None
