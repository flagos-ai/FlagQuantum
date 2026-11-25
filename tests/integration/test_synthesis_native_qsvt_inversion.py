"""Phase-exact native QSVT composition, independent matrices and original mapping."""

from pathlib import Path

import pytest
import torch

import flagquantum as fq
from examples.synthesis.algorithms.quantum_linear_solvers.qsvt_matrix_inversion.comparisons.circuit import (
    reference_circuit,
)
from examples.synthesis.algorithms.quantum_linear_solvers.qsvt_matrix_inversion.data import (
    AMPLITUDES,
    MATRIX,
    PHASES,
)
from examples.synthesis.algorithms.quantum_linear_solvers.qsvt_matrix_inversion.main import (
    build_program,
    main,
    transform,
)
from flagquantum import synth
from flagquantum.errors import PlanningError
from flagquantum.synth import (
    Output,
    QArray,
    QBit,
    adjoint,
    allocate,
    qfunc,
)
from tests.integration.test_synthesis_qmod_qsvt_inversion import source_literals

pytestmark = pytest.mark.integration
ORIGINAL = (
    Path(__file__).resolve().parents[1]
    / "fixtures/classiq_qmod/corpus/original/algorithms/quantum_linear_solvers/qsvt_matrix_inversion/qsvt_matrix_inversion.qmod"
)


@pytest.fixture(scope="module")
def accepted_program() -> synth.Program:
    return build_program()


def full_state(program: synth.Program, expected: torch.Tensor) -> torch.Tensor:
    raw = fq.run(program).statevector()
    assert isinstance(raw, torch.Tensor)
    state = raw.reshape(16, -1)
    torch.testing.assert_close(state[:, 0], expected.flatten(), atol=2e-10, rtol=0)
    assert float(state[:, 1:].abs().square().sum()) < 1e-22
    return state[:, 0]


def test_original_literals_bit_mapping_and_linear_solution(
    accepted_program: synth.Program,
) -> None:
    matrix, rhs, phases = source_literals(ORIGINAL)
    assert phases == PHASES and len(PHASES) == 70
    torch.testing.assert_close(
        matrix, torch.tensor(MATRIX, dtype=torch.complex128), atol=0, rtol=0
    )
    torch.testing.assert_close(
        rhs, torch.tensor(AMPLITUDES, dtype=torch.complex128), atol=0, rtol=0
    )
    expected = fq.run(reference_circuit()).statevector()
    actual = full_state(accepted_program, expected)
    original = synth.synthesize(ORIGINAL)
    raw = fq.run(original).statevector().reshape(16, -1)
    # Source wire order is block,state MSB,state LSB,signal auxiliary.
    remapped = raw[:, 0].reshape(2, 4, 2).permute(1, 0, 2).flatten()
    torch.testing.assert_close(actual, remapped, atol=2e-10, rtol=0)
    assert float(raw[:, 1:].abs().square().sum()) < 1e-22
    solution = (
        torch.linalg.solve(matrix[:4, :4], rhs)
        * torch.linalg.svdvals(matrix[:4, :4])[-1]
        / 2
    )
    torch.testing.assert_close(
        actual.reshape(4, 2, 2)[:, 0, 0], solution, atol=1e-7, rtol=0
    )
    assert accepted_program.registers == {"state": (0, 1), "block": (2,), "aux": (3,)}


@pytest.mark.parametrize("perturbation", [0.37, -0.61])
def test_nonzero_phase_variants_preserve_absolute_full_state(
    perturbation: float,
) -> None:
    phases = tuple(
        angle + perturbation * (index + 1) / 70 for index, angle in enumerate(PHASES)
    )
    full_state(build_program(phases), fq.run(reference_circuit(phases)).statevector())


@qfunc
def inverse_main(
    phases: list[float],
    state: Output[QArray[QBit]],
    block: Output[QBit],
    aux: Output[QBit],
) -> None:
    allocate(2, state)
    allocate(block)
    allocate(aux)
    with adjoint():
        transform(phases, state, block, aux)


def test_checked_inverse_restores_input_and_all_work() -> None:
    # Each compiled block stays within the authoritative 100,000-operation cap.
    # Execute the native inverse on the independent full forward state.
    program = synth.synthesize(inverse_main, classical_inputs={"phases": list(PHASES)})
    forward = fq.run(reference_circuit()).statevector().flatten()
    inputs = torch.zeros(2**program.circuit.n_wires, dtype=torch.complex128)
    stride = inputs.numel() // 16
    inputs[::stride] = forward
    actual = (
        fq.Circuit.from_ir(program.circuit, dtype=torch.complex128, inputs=inputs)
        .state()
        .reshape(16, -1)
    )
    expected = torch.zeros((4, 2, 2), dtype=torch.complex128)
    rhs = torch.tensor(AMPLITUDES, dtype=torch.complex128)
    expected[:, 0, 0] = rhs / torch.linalg.vector_norm(rhs)
    torch.testing.assert_close(actual[:, 0], expected.flatten(), atol=2e-10, rtol=0)
    assert float(actual[:, 1:].abs().square().sum()) < 1e-22


@pytest.mark.parametrize(
    "kind", ["qubits", "workspace", "operations", "depth", "memory"]
)
def test_resource_budget_refusals(kind: str) -> None:
    inputs = {"phases": list(PHASES), "amplitudes": list(AMPLITUDES)}
    with pytest.raises((synth.SynthesisError, PlanningError)):
        if kind == "qubits":
            synth.synthesize(main, classical_inputs=inputs, max_qubits=3)
        elif kind == "workspace":
            synth.synthesize(main, classical_inputs=inputs, max_workspace=0)
        elif kind == "operations":
            synth.synthesize(main, classical_inputs=inputs, max_operations=1)
        elif kind == "depth":
            synth.synthesize(main, classical_inputs=inputs, max_depth=1)
        else:
            synth.synthesize(main, classical_inputs=inputs, memory_limit_bytes=16)


def test_incomplete_phase_sequence_refuses() -> None:
    with pytest.raises(ValueError, match="all 70 phases"):
        build_program(PHASES[:-1])
