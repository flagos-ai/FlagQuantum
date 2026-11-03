"""Execution and persistence contracts for register-indexed Majorana selection."""

import json

import pytest
import torch

from flagquantum import Circuit
from flagquantum.algorithms.primitives.multiplexers import SelectedMajoranaFermion
from flagquantum.core.ir import CircuitIR
from flagquantum.errors import ValidationError


@pytest.mark.parametrize("target_gate", ["X", "Y", "Z"])
def test_coherent_selection_survives_ir_round_trip(target_gate: str) -> None:
    dtype = torch.complex64
    inputs = torch.eye(64, dtype=dtype)
    circuit = Circuit(6, bsz=len(inputs), inputs=inputs)
    SelectedMajoranaFermion.make_on(
        selection=(1, 2), target=(3, 4, 5), control=(0,), target_gate=target_gate
    ).append_to(circuit)

    identity = torch.eye(2, dtype=dtype)
    z = torch.diag(torch.tensor([1, -1], dtype=dtype))
    pauli = {
        "X": torch.tensor([[0, 1], [1, 0]], dtype=dtype),
        "Y": torch.tensor([[0, -1j], [1j, 0]], dtype=dtype),
        "Z": z,
    }[target_gate]
    expected = torch.block_diag(
        torch.eye(32, dtype=dtype),
        torch.kron(torch.kron(pauli, identity), identity),
        torch.kron(torch.kron(z, pauli), identity),
        torch.kron(torch.kron(z, z), pauli),
        torch.eye(8, dtype=dtype),
    )
    restored = CircuitIR.from_dict(json.loads(json.dumps(circuit.to_ir().to_dict())))
    assert restored.instructions[0].params == {
        "selection": (1, 2),
        "target": (3, 4, 5),
        "control": (0,),
        "target_gate": target_gate,
    }
    torch.testing.assert_close(circuit.state(), expected.T)
    coherent = torch.arange(1, 65, dtype=torch.float32).to(dtype)
    coherent /= coherent.norm()
    torch.testing.assert_close(
        Circuit.from_ir(restored, inputs=coherent).state().reshape(-1),
        expected @ coherent,
    )


def test_invalid_binding_leaves_circuit_unchanged() -> None:
    circuit = Circuit(2)
    operation = SelectedMajoranaFermion.make_on(selection=(0,), target=(0, 1))
    with pytest.raises(ValidationError, match="disjoint"):
        operation.append_to(circuit)
    assert not circuit.to_ir().instructions
