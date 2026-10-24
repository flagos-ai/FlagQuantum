"""Simultaneous Pauli responses checked against independent dense evolution."""

from __future__ import annotations

from itertools import product

import pytest
import torch

import flagquantum as fq
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.errors import CapabilityError
from flagquantum.noise import (
    KrausChannel,
    amplitude_damping_channel,
    two_qubit_depolarizing_channel,
)
from flagquantum.simulation.stabilizer._fault_signatures import (
    _clifford_product_signatures,
    _pauli_channel_terms,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("gate", "inverse", "targets"),
    [
        ("i", "i", (0,)),
        ("x", "x", (0,)),
        ("y", "y", (1,)),
        ("z", "z", (0,)),
        ("sdg", "s", (1,)),
        ("sxdg", "sx", (1,)),
        ("cy", "cy", (0, 1)),
        ("cz", "cz", (1, 0)),
        ("h", "h", (0,)),
        ("s", "sdg", (1,)),
        ("sx", "sxdg", (0,)),
        ("cx", "cx", (0, 1)),
        ("cx", "cx", (1, 0)),
        ("cy", "cy", (1, 0)),
        ("cz", "cz", (0, 1)),
        ("swap", "swap", (1, 0)),
    ],
)
def test_every_product_and_position_matches_dense_evolution(
    gate: str, inverse: str, targets: tuple[int, ...]
) -> None:
    instructions = (
        Instruction("x", (0,)),
        Instruction("h", (0,)),
        Instruction("sx", (1,)),
        Instruction(gate, targets),
        Instruction(inverse, targets),
        Instruction("sxdg", (1,)),
        Instruction("h", (0,)),
    )
    faults = tuple(
        (position, (1, 0), "".join(letters))
        for position in range(len(instructions))
        for letters in product("IXYZ", repeat=2)
    )
    responses = _clifford_product_signatures(
        CircuitIR(n_wires=2, instructions=instructions),
        parities=(((), (0,)), ((), (1,)), ((), (0, 1))),
        faults=faults,
    )
    for (position, qubits, labels), response in zip(faults, responses, strict=True):
        injected = tuple(
            Instruction(label.lower(), (qubit,))
            for qubit, label in zip(qubits, labels, strict=True)
            if label != "I"
        )
        faulty = CircuitIR(
            n_wires=2,
            instructions=(
                *instructions[: position + 1],
                *injected,
                *instructions[position + 1 :],
            ),
        )
        probabilities = fq.Circuit.from_ir(faulty).state().abs().square().reshape(2, 2)
        bits = [int(probabilities.sum(dim=1 - q).argmax()) for q in (0, 1)]
        assert float(probabilities[bits[0], bits[1]]) == pytest.approx(1, abs=1e-6)
        assert response == bits[0] | bits[1] << 1 | (bits[0] ^ bits[1]) << 2


def test_products_preserve_measurement_cancellation_and_reset_erasure() -> None:
    program = CircuitIR(
        n_wires=2,
        instructions=(
            Instruction("h", (0,)),
            Instruction("measure", (0,), metadata={"is_dynamic": True}),
            Instruction("measure", (0,), metadata={"is_dynamic": True}),
            Instruction("reset", (1,), metadata={"is_dynamic": True}),
            Instruction("x", (1,)),
        ),
    )
    assert _clifford_product_signatures(
        program,
        parities=(((0, 1), ()), ((), (1,))),
        faults=((0, (0, 1), "XX"), (1, (0, 1), "YX"), (3, (0, 1), "IX")),
    ) == (2, 3, 0)


@pytest.mark.parametrize("probability", [0.0, 0.2, 1.0])
def test_depolarization_has_all_fifteen_products(probability: float) -> None:
    terms = _pauli_channel_terms(two_qubit_depolarizing_channel(probability))
    assert tuple(label for label, _ in terms) == (
        tuple("".join(p) for p in product("IXYZ", repeat=2) if p != ("I", "I"))
        if probability
        else ()
    )
    assert all(
        weight == pytest.approx(probability / 15, abs=1e-7) for _, weight in terms
    )


def test_duplicate_kraus_branches_and_global_phases_are_classified_once() -> None:
    identity = torch.eye(2, dtype=torch.complex128)
    x = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128)
    channel = KrausChannel(
        "biased",
        (
            (0.5**0.5) * identity,
            (0.2**0.5) * x,
            1j * (0.3**0.5) * x,
            torch.zeros_like(x),
        ),
    )
    terms = _pauli_channel_terms(channel)
    assert len(terms) == 1 and terms[0][0] == "X"
    assert terms[0][1] == pytest.approx(0.5)


def test_non_pauli_and_large_channels_are_refused() -> None:
    coherent = torch.tensor(
        [[1, 0], [0, complex(2**-0.5, 2**-0.5)]], dtype=torch.complex128
    )
    for channel in (
        amplitude_damping_channel(0.2),
        KrausChannel("coherent", (coherent,)),
        KrausChannel("three_qubits", (torch.eye(8, dtype=torch.complex128),)),
    ):
        with pytest.raises(CapabilityError):
            _pauli_channel_terms(channel)
