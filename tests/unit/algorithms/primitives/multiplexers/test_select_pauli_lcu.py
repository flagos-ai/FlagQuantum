"""Exercise deferred Pauli table selection on remapped registers."""

import json

import pytest
import torch

import flagquantum as fq
from flagquantum.algorithms.primitives.multiplexers import SelectPauliLCU
from flagquantum.compiler.lowering.deferred_operations import (
    lower_select_pauli_lcu_to_gates,
)
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


@pytest.mark.parametrize(
    ("selection_width", "control_val", "words"),
    [
        (0, None, ("-Y",)),
        (0, 0, ("-I",)),
        (2, None, ("X", "-Y", "Z")),
        (3, 0, ("-I", "X", "-Y", "Z")),
        (3, 1, ("-I", "X", "-Y", "Z")),
    ],
)
def test_gate_lowering_matches_dense_with_clean_workspace(
    selection_width: int, control_val: int | None, words: tuple[str, ...]
) -> None:
    control_width = int(control_val is not None)
    width = control_width + selection_width + 1
    order = (width - 1, *range(width - 1))
    generator = torch.Generator().manual_seed(42)
    state = torch.randn(2**width, dtype=torch.complex128, generator=generator)
    state /= torch.linalg.vector_norm(state)
    dense = fq.Circuit(width, dtype=state.dtype, inputs=state)
    SelectPauliLCU(selection_width, 1, words, control_val).append_to(
        dense,
        control=order[:control_width],
        selection=order[control_width:-1],
        target=order[-1:],
    )
    instruction = dense.to_ir().instructions[0]
    lowered = lower_select_pauli_lcu_to_gates(instruction)
    assert {gate.name for gate in lowered.to_ir().instructions} <= {
        "x",
        "y",
        "z",
        "h",
        "s",
        "sdg",
        "cx",
        "ccx",
    }
    scratch = tuple(range(width, lowered.n_qubits))
    initial = torch.zeros((state.numel(), 2 ** len(scratch)), dtype=state.dtype)
    initial[:, 0] = state
    circuit = fq.Circuit(lowered.n_qubits, dtype=state.dtype, inputs=initial.flatten())
    circuit.compose(lowered, qubits=(*instruction.wires, *scratch))
    result = circuit.state().reshape_as(initial)
    torch.testing.assert_close(
        result[:, 0], dense.state().flatten(), atol=1e-12, rtol=0
    )
    torch.testing.assert_close(
        result[:, 1:], torch.zeros_like(result[:, 1:]), atol=1e-12, rtol=0
    )
