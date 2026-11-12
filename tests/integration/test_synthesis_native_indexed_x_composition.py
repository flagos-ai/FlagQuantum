"""Nested predicate XOR keeps arbitrary markers, branch phases and work limits."""

import cmath

import pytest
import torch

import flagquantum as fq
from flagquantum import synth
from flagquantum.compiler import ResourceEstimate
from flagquantum.compiler._construction.constraints import CircuitLimits
from flagquantum.compiler._construction.frontend_python import _QuantumFunction
from flagquantum.compiler._construction.register_lowering import (
    RegisterCandidate,
    schedule_register_program,
)
from flagquantum.compiler._construction.selection import (
    Constraints,
    select_register_program,
)
from tests.integration.test_synthesis_native_exact_numbers import definition
from tests.register_program_cases import Registers

pytestmark = pytest.mark.integration


def entry(marker: int, inverse: bool = False) -> _QuantumFunction:
    helpers = """
@qfunc
def mark(word: QNum, target: QBit) -> None:
    control((word == 0) | (word == 31), lambda: X(target))
    phase(1.0, 0.23)
@qfunc
def inner(c: QBit, word: QNum, target: QBit) -> None:
    control(c, lambda: mark(word, target))
"""
    operation = "control(d == 0, lambda: inner(c, word, target))"
    if inverse:
        operation = "with adjoint():\n    " + operation
    preparation = (
        (
            "X(target)"
            if marker == 1
            else "R(0.73, 1.5707963267948966, target)\nRZ(0.41, target)"
        )
        if marker
        else ""
    )
    return definition(
        "d:Output[QBit], c:Output[QBit], word:Output[QNum[5]], target:Output[QBit]",
        "allocate(d)\nallocate(c)\nallocate(word)\nallocate(target)\n"
        "H(d)\nH(c)\nhadamard_transform(word)\nRZ(0.17,d)\nRZ(-0.31,c)\nRZ(0.19,word[0])\n"
        + preparation
        + "\n"
        + operation,
        helpers,
    )


def reference(marker: int, inverse: bool) -> fq.Circuit:
    circuit = fq.Circuit(8, dtype=torch.complex128)
    for wire in range(7):
        circuit.h(wire)
    circuit.rz(0, 0.17).rz(1, -0.31).rz(6, 0.19)
    if marker == 1:
        circuit.x(7)
    elif marker == 2:
        circuit.ry(7, 0.73).rz(7, 0.41)
    matrix = torch.zeros((256, 256), dtype=torch.complex128)
    for column in range(256):
        d, c, word = column >> 7, (column >> 6) & 1, (column >> 1) & 31
        active = d == 0 and c == 1
        row = column ^ int(active and word in (0, 31))
        matrix[row, column] = torch.tensor(
            cmath.exp(((-1 if inverse else 1) * 0.23j) * active), dtype=torch.complex128
        )
    return circuit.any(*range(8), matrix=matrix)


@pytest.mark.parametrize("marker", [0, 1, 2])
@pytest.mark.parametrize("inverse", [False, True])
@pytest.mark.parametrize("workspace", [0, 6])
def test_nested_masks_keep_absolute_phase_and_arbitrary_marker(
    marker: int, inverse: bool, workspace: int
) -> None:
    program = synth.synthesize(entry(marker, inverse), max_workspace=workspace)
    actual = fq.run(program).statevector().reshape(256, -1)
    torch.testing.assert_close(
        actual[:, 0], reference(marker, inverse).state().flatten(), atol=1e-11, rtol=0
    )
    assert float(actual[:, 1:].abs().square().sum()) < 1e-22
    used_workspace = program.report["workspace_qubits"]
    assert isinstance(used_workspace, int)
    assert used_workspace <= workspace


def test_clean_ladder_saves_actual_gates_and_keeps_scratch_free_alternative() -> None:
    source = entry(2)
    narrow = synth.synthesize(source, max_workspace=0)
    fast = synth.synthesize(source, max_workspace=6)
    assert narrow.report["workspace_qubits"] == 0
    fast_workspace = fast.report["workspace_qubits"]
    assert isinstance(fast_workspace, int)
    assert 0 < fast_workspace <= 6
    narrow_cost = narrow.report["resources"]
    fast_cost = fast.report["resources"]
    assert isinstance(narrow_cost, ResourceEstimate)
    assert isinstance(fast_cost, ResourceEstimate)
    assert fast_cost.n_operations < narrow_cost.n_operations
    with pytest.raises(ValueError, match="operation|budget"):
        synth.synthesize(
            source, max_workspace=6, max_operations=fast_cost.n_operations - 1
        )


@pytest.mark.parametrize("workspace", [None, 0, 1, 3, 256])
def test_masked_x_keeps_grouped_and_scratch_free_budget_alternatives(
    workspace: int | None,
) -> None:
    model = Registers()
    word = model.declare("word", 5)
    target = model.declare("target", 1)
    model.emit("qreg.allocate", word)
    model.emit("qreg.allocate", target)
    for bit in range(5):
        model.emit("qreg.gate", word, gate="h", index=bit)
    model.emit("qreg.gate", word, gate="rz", index=0, angle=0.19)
    model.emit("qreg.gate", target, gate="ry", index=0, angle=0.73)
    model.emit("qreg.gate", target, gate="rz", index=0, angle=0.41)
    model.emit(
        "qreg.control_x",
        *(word for _ in range(5)),
        target,
        indices=(0, 1, 2, 3, 4, 0),
        control_state=0b10100,
    )
    if workspace is None or workspace == 256:
        selection = select_register_program(
            model.finish(), constraints=Constraints(CircuitLimits(max_qubits=workspace))
        )
        assert selection.selected is not None
        assert isinstance(selection.selected.arithmetic, RegisterCandidate)
        plan = selection.selected.arithmetic
    else:
        plan = schedule_register_program(
            model.finish(), n_qubits=9, _clean_qubits=6 + workspace
        )
    reference = fq.Circuit(6, dtype=torch.complex128)
    for wire in range(5):
        reference.h(wire)
    reference.rz(4, 0.19).ry(5, 0.73).rz(5, 0.41)
    permutation = torch.eye(64, dtype=torch.complex128)
    permutation[[40, 41]] = permutation[[41, 40]]
    reference.any(*range(6), matrix=permutation)
    actual = fq.run(fq.Circuit.from_ir(plan.circuit)).statevector().reshape(64, -1)
    torch.testing.assert_close(
        actual[:, 0], reference.state().flatten(), atol=1e-11, rtol=0
    )
    assert float(actual[:, 1:].abs().square().sum()) < 1e-22
    assert plan.peak_workspace == (
        1 if workspace is None or workspace == 256 else workspace
    )
