"""Fault responses checked against dense two-qubit evolution and measured identities."""

from __future__ import annotations

import pytest
import torch

import flagquantum as fq
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.errors import CapabilityError
from flagquantum.simulation.stabilizer._fault_signatures import (
    _clifford_fault_signatures,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("gate", "inverse", "wires"),
    [
        ("i", "i", (0,)),
        ("x", "x", (0,)),
        ("y", "y", (0,)),
        ("z", "z", (0,)),
        ("h", "h", (0,)),
        ("s", "sdg", (0,)),
        ("sdg", "s", (0,)),
        ("sx", "sxdg", (0,)),
        ("sxdg", "sx", (0,)),
        ("cx", "cx", (1, 0)),
        ("cy", "cy", (0, 1)),
        ("cz", "cz", (1, 0)),
        ("swap", "swap", (0, 1)),
    ],
)
def test_every_clifford_and_pauli_fault_matches_dense_evolution(
    gate: str, inverse: str, wires: tuple[int, ...]
) -> None:
    prelude = (Instruction("h", (0,)), Instruction("sx", (1,)))
    instructions = (
        *prelude,
        Instruction(gate, wires),
        Instruction(inverse, wires),
        Instruction("sxdg", (1,)),
        Instruction("h", (0,)),
    )
    program = CircuitIR(n_wires=2, instructions=instructions)
    faults = tuple(
        (position, wire, pauli)
        for position in range(len(instructions))
        for wire in (0, 1)
        for pauli in ("X", "Y", "Z")
    )
    responses = _clifford_fault_signatures(
        program, parities=(((), (0,)), ((), (1,)), ((), (0, 1))), faults=faults
    )
    for (position, wire, pauli), response in zip(faults, responses, strict=True):
        faulty = CircuitIR(
            n_wires=2,
            instructions=(
                *instructions[: position + 1],
                Instruction(pauli.lower(), (wire,)),
                *instructions[position + 1 :],
            ),
        )
        probabilities = fq.Circuit.from_ir(faulty).state().abs().square().reshape(2, 2)
        bits = []
        for qubit in (0, 1):
            marginal = probabilities.sum(dim=1 - qubit)
            bit = int(marginal.argmax())
            assert torch.allclose(marginal[bit], torch.tensor(1.0), atol=1e-6)
            bits.append(bit)
        expected = bits[0] | (bits[1] << 1) | ((bits[0] ^ bits[1]) << 2)
        assert response == expected, (gate, position, wire, pauli)


def test_random_measurements_cancel_in_a_record_parity() -> None:
    program = CircuitIR(
        n_wires=1,
        instructions=(
            Instruction("h", (0,)),
            Instruction("measure", (0,), metadata={"is_dynamic": True}),
            Instruction("measure", (0,), metadata={"is_dynamic": True}),
        ),
    )
    assert _clifford_fault_signatures(
        program,
        parities=(((0, 1), ()),),
        faults=((0, 0, "X"), (1, 0, "X"), (1, 0, "Z"), (2, 0, "X")),
    ) == (0, 1, 0, 0)
    with pytest.raises(ValueError, match="not deterministic"):
        _clifford_fault_signatures(program, parities=(((0,), ()),), faults=())


def test_reset_erases_prior_faults_and_odd_baseline_is_preserved() -> None:
    program = CircuitIR(
        n_wires=1,
        instructions=(
            Instruction("x", (0,)),
            Instruction("reset", (0,), metadata={"is_dynamic": True}),
            Instruction("x", (0,)),
            Instruction("measure", (0,), metadata={"is_dynamic": True}),
        ),
    )
    assert _clifford_fault_signatures(
        program,
        parities=(((0,), ()),),
        faults=((0, 0, "X"), (1, 0, "X"), (2, 0, "Z"), (3, 0, "X")),
    ) == (1, 0, 1, 1)


@pytest.mark.parametrize(
    "instruction",
    [
        Instruction("t", (0,)),
        Instruction("h", (0,), metadata={"conditions": ((0, 1),)}),
        Instruction("h", (0,), matrix=((1, 0), (0, 1))),
        Instruction("bit_flip", (0,), params={"probability": 0.1}),
    ],
)
def test_unsupported_programs_fail_closed(instruction: Instruction) -> None:
    with pytest.raises(CapabilityError):
        _clifford_fault_signatures(
            CircuitIR(n_wires=1, instructions=(instruction,)),
            parities=(((), (0,)),),
            faults=(),
        )


@pytest.mark.parametrize("fault", [(-1, 0, "X"), (0, 1, "Z"), (0, 0, "H")])
def test_invalid_fault_locations_are_refused(fault: tuple[int, int, str]) -> None:
    with pytest.raises(ValueError):
        _clifford_fault_signatures(
            CircuitIR(n_wires=1, instructions=(Instruction("i", (0,)),)),
            parities=(((), (0,)),),
            faults=(fault,),
        )
