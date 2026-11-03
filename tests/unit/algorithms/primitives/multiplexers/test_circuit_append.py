"""Test the bound-operation construction and deferred execution contract."""

import json

import pytest
import torch

import flagquantum as fq
from flagquantum.algorithms.primitives.multiplexers import (
    SelectedMajoranaFermion,
    SelectPauliLCU,
)
from flagquantum.core.ir import CircuitIR

pytestmark = pytest.mark.unit


def test_append_select_operations_round_trip_and_execute() -> None:
    circuit = fq.Circuit(3).x(0)
    majorana = SelectedMajoranaFermion.make_on(selection=[0], target=[1, 2])
    select = SelectPauliLCU(1, 2, ("II", "XI")).make_on(selection=[0], target=[1, 2])
    assert circuit.append(majorana).append(select) is circuit

    ir = CircuitIR.from_dict(json.loads(json.dumps(circuit.to_ir().to_dict())))
    assert [op.name for op in ir.instructions[1:]] == [
        "selected_majorana_fermion",
        "select_pauli_lcu",
    ]
    assert all(op.matrix is None for op in ir.instructions[1:])
    expected = torch.zeros((1, 8), dtype=torch.complex64)
    expected[0, 7] = 1j
    torch.testing.assert_close(fq.Circuit.from_ir(ir).state(), expected)


def test_append_rejects_an_unsupported_value() -> None:
    circuit = fq.Circuit(3)
    with pytest.raises(TypeError, match="bound circuit operation"):
        circuit.append(object())  # type: ignore[arg-type]
    assert not circuit.to_ir().instructions
