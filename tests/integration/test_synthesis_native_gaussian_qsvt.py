"""Gaussian LCU full states, approved overlap, absolute phase and clean ports."""

import ast
import math
import re
from pathlib import Path
from typing import TypedDict

import pytest
import torch

import flagquantum as fq
from examples.synthesis.community.paper_implementation_project.quantum_state_preparation_without_coherent_arithmetic.stateprep_guassian_using_qsvt import (
    main as native,
)
from examples.synthesis.community.paper_implementation_project.quantum_state_preparation_without_coherent_arithmetic.stateprep_guassian_using_qsvt.comparisons.circuit import (
    direct_sum,
    preparation_gates,
    reference_circuit,
    replay,
    rotation,
)
from examples.synthesis.community.paper_implementation_project.quantum_state_preparation_without_coherent_arithmetic.stateprep_guassian_using_qsvt.data import (
    EVEN_PHASES,
    ODD_PHASES,
    SINE_ANGLES,
    SUCCESS_AMPLITUDE,
)
from flagquantum import synth
from flagquantum.compiler._hybrid.verifier import HybridVerificationError
from flagquantum.ecosystem.contracts import InteropConversionError
from flagquantum.errors import PlanningError
from flagquantum.synth import (
    Output,
    QArray,
    QBit,
    R,
    adjoint,
    allocate,
    hadamard_transform,
    phase,
    qfunc,
)


class Limits(TypedDict, total=False):
    max_qubits: int
    max_workspace: int
    max_operations: int
    max_depth: int
    memory_limit_bytes: int


pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[2]
RELATIVE = "community/paper_implementation_project/quantum_state_preparation_without_coherent_arithmetic/stateprep_guassian_using_qsvt.qmod"
FIXTURES = ROOT / "tests/fixtures/classiq_qmod/corpus"


def state(program: synth.Program, size: int) -> torch.Tensor:
    result = fq.run(program).statevector()
    assert isinstance(result, torch.Tensor)
    viewed = result.reshape(size, -1)
    assert float(viewed[:, 1:].abs().square().sum()) < 1e-22
    return viewed[:, 0]


def test_complete_native_approved_and_independent_circuit() -> None:
    expected = fq.run(reference_circuit()).statevector().reshape(2048, 2, -1)
    assert float(expected[:, 1:].abs().square().sum()) < 1e-22
    assert float(expected[:, :, 1:].abs().square().sum()) < 1e-22
    program = native.build_program()
    actual = state(program, 8192).reshape(2048, 4)
    assert float(actual[:, 1:].abs().square().sum()) < 1e-22
    torch.testing.assert_close(actual[:, 0], expected[:, 0, 0], atol=1e-10, rtol=0)
    approved = synth.synthesize(FIXTURES / "approved" / RELATIVE)
    torch.testing.assert_close(
        state(approved, 2048), expected[:, 0, 0], atol=1e-10, rtol=0
    )
    assert len(program.registers["data"]) == 8
    assert program.report["workspace_qubits"] == 5


@qfunc
def prepared(
    even: list[float],
    odd: list[float],
    angles: list[float],
    data: Output[QArray[QBit]],
    signal: Output[QBit],
    aux: Output[QBit],
    selector: Output[QBit],
) -> None:
    allocate(8, data)
    allocate(signal)
    allocate(aux)
    allocate(selector)
    native.prepare(even, odd, angles, data, signal, aux, selector)


def test_preparation_overlap_and_retained_model() -> None:
    source = (FIXTURES / "original" / RELATIVE).read_text()
    match = re.search(r"qsvt_lcu\((\[.*?\]), (\[.*?\]), lambda", source, re.S)
    assert match
    assert tuple(ast.literal_eval(match[1])) == EVEN_PHASES and len(EVEN_PHASES) == 29
    assert tuple(ast.literal_eval(match[2])) == ODD_PHASES and len(ODD_PHASES) == 28
    assert tuple(2.0 ** (-7 + i) for i in range(8)) == SINE_ANGLES
    inputs = native.classical_inputs()
    inputs.pop("beta")
    actual = state(synth.synthesize(prepared, classical_inputs=inputs), 2048)
    reference = fq.Circuit(11, dtype=torch.complex128)
    replay(reference, preparation_gates())
    torch.testing.assert_close(
        actual, fq.run(reference).statevector().flatten(), atol=1e-11, rtol=0
    )
    observed = math.sqrt(float(actual.reshape(256, 8)[:, 0].abs().square().sum()))
    assert abs(observed - SUCCESS_AMPLITUDE) < 1e-13
    with pytest.raises(InteropConversionError, match="supplied amplitude.*disagrees"):
        synth.synthesize(FIXTURES / "original" / RELATIVE)


@qfunc
def selected(
    angles: list[float],
    phi: float,
    undo: int,
    data: Output[QArray[QBit]],
    signal: Output[QBit],
    selector: Output[QBit],
) -> None:
    allocate(8, data)
    allocate(signal)
    allocate(selector)
    hadamard_transform(data)
    R(0.73, math.pi / 2, signal)
    R(1.19, math.pi / 2, selector)
    phase(signal, phi)
    phase(selector, -phi)
    native.selected_back(angles, data, signal, selector)
    if undo == 1:
        with adjoint():
            native.selected_back(angles, data, signal, selector)


@pytest.mark.parametrize("scale,phi", [(0.8, 0.37), (1.1, -0.61)])
@pytest.mark.parametrize("undo", [False, True])
def test_selected_inverse_on_nonzero_complex_states(
    scale: float, phi: float, undo: bool
) -> None:
    angles = [scale * angle for angle in SINE_ANGLES]
    program = synth.synthesize(
        selected, classical_inputs={"angles": angles, "phi": phi, "undo": int(undo)}
    )
    circuit = fq.Circuit(10, dtype=torch.complex128)
    for wire in range(8):
        circuit.h(wire)
    circuit.ry(8, theta=0.73).ry(9, theta=1.19).phase(8, theta=phi).phase(9, theta=-phi)
    if not undo:
        x = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128)
        circuit.any(9, 8, matrix=direct_sum(x, torch.eye(2, dtype=torch.complex128)))
        for bit, angle in enumerate(angles):
            controlled_y = direct_sum(
                torch.eye(2, dtype=torch.complex128), rotation(-angle)
            )
            circuit.any(
                9,
                7 - bit,
                8,
                matrix=direct_sum(controlled_y, torch.eye(4, dtype=torch.complex128)),
            )
    torch.testing.assert_close(
        state(program, 1024),
        fq.run(circuit).statevector().flatten(),
        atol=1e-11,
        rtol=0,
    )


@qfunc
def aliased(
    angles: list[float], data: Output[QArray[QBit]], signal: Output[QBit]
) -> None:
    allocate(8, data)
    allocate(signal)
    native.selected_back(angles, data, signal, signal)


def test_cross_call_alias_refuses() -> None:
    with pytest.raises(HybridVerificationError, match="disjoint"):
        synth.synthesize(aliased, classical_inputs={"angles": list(SINE_ANGLES)})


@pytest.mark.parametrize(
    "amplitude", [0.39579583583036, 0.0, 1.0, float("nan"), float("inf"), True]
)
def test_invalid_declared_overlap_refuses(amplitude: float) -> None:
    with pytest.raises(ValueError, match="amplitude"):
        native.build_program(amplitude)


@pytest.mark.parametrize(
    "limits",
    [
        {"max_qubits": 17},
        {"max_workspace": 4},
        {"max_operations": 3756},
        {"max_depth": 3185},
        {"memory_limit_bytes": 1024},
    ],
)
def test_resource_budgets_refuse(limits: Limits) -> None:
    with pytest.raises(
        (PlanningError, HybridVerificationError),
        match="budget|limit|requires|exceed|infeasible",
    ):
        synth.synthesize(
            native.main, classical_inputs=native.classical_inputs(), **limits
        )
