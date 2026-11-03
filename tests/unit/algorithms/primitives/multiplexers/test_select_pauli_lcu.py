"""Exercise deferred Pauli table selection on remapped registers."""

import json

import pytest
import torch

import flagquantum as fq
from flagquantum.algorithms.primitives.multiplexers import SelectPauliLCU
from flagquantum.core.ir import CircuitIR

pytestmark = pytest.mark.unit


def test_partial_table_roundtrip_and_composition() -> None:
    component = SelectPauliLCU(1, 1, ("-Y",), control_val=0)
    local = component.append_to(
        fq.Circuit(3), control=(0,), selection=(1,), target=(2,)
    )
    circuit = fq.Circuit(3).h(0).compose(local, qubits=(2, 0, 1))
    ir = CircuitIR.from_dict(json.loads(json.dumps(circuit.to_ir().to_dict())))
    assert ir.instructions[-1].wires == (2, 0, 1)
    assert ir.instructions[-1].matrix is None
    expected = torch.zeros((1, 8), dtype=torch.complex64)
    expected[0, 2] = -1j / 2**0.5
    expected[0, 4] = 1 / 2**0.5
    torch.testing.assert_close(fq.Circuit.from_ir(ir).state(), expected)
