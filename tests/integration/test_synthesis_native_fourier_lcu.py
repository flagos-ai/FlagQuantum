"""Original Fourier LCU preserves its complete controller/data environment."""

from pathlib import Path

import pytest
import torch

import flagquantum as fq
from examples.synthesis.tutorials.basics.quantum_primitives.linear_combination_of_unitaries.comparisons.circuit import (
    lcu_matrix,
    reference_circuit,
)
from examples.synthesis.tutorials.basics.quantum_primitives.linear_combination_of_unitaries.main import (
    build_program,
    main,
)
from examples.synthesis.tutorials.basics.quantum_primitives.linear_combination_of_unitaries.physics import (
    PROBABILITIES,
    ZERO_INPUT,
)
from flagquantum import synth
from flagquantum.compiler import ResourceEstimate
from flagquantum.compiler._hybrid.verifier import HybridVerificationError
from flagquantum.errors import PlanningError

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[2]


def assert_complete(program: synth.Program, expected: torch.Tensor) -> None:
    work = program.report["workspace_qubits"]
    assert isinstance(work, int)
    complete = torch.zeros((16, 1 << work), dtype=torch.complex128)
    complete[:, 0] = expected.flatten()
    result = fq.run(program)
    torch.testing.assert_close(
        result.statevector().flatten(), complete.flatten(), atol=1e-11, rtol=0
    )
    assert (
        float(result.statevector().reshape(16, -1)[:, 1:].abs().square().sum()) < 1e-22
    )
    resources = program.report["resources"]
    assert isinstance(resources, ResourceEstimate)
    assert resources.n_qubits == 4 + work
    probabilities = program.probabilities(result, registers=("controller", "psi"))
    for c in range(4):
        for q in range(4):
            assert probabilities.get((c, q), 0) == pytest.approx(
                float(expected.flatten()[4 * c + q].abs().square()), abs=1e-11, rel=0
            )


def test_unchanged_source_native_and_independent_full_states() -> None:
    original = (
        ROOT
        / "tests/fixtures/classiq_qmod/corpus/original/tutorials/basic_tutorials/quantum_primitives/linear_combination_of_unitaries/linear_combination_of_unitaries.qmod"
    )
    expected = fq.run(reference_circuit()).statevector()
    for program in (synth.synthesize(original), build_program()):
        assert_complete(program, expected)


@pytest.mark.parametrize("variant", [0, 1])
def test_two_nonzero_weight_and_coherent_data_models(variant: int) -> None:
    probabilities = ((0.13, 0.21, 0.29, 0.37), (0.41, 0.17, 0.23, 0.19))[variant]
    values = ((0.3, -0.4, 0.5, -0.7), (-0.2, 0.6, 0.4, 0.5))[variant]
    norm = sum(v * v for v in values) ** 0.5
    amplitudes = tuple(v / norm for v in values)
    assert_complete(
        build_program(probabilities, amplitudes),
        fq.run(reference_circuit(probabilities, amplitudes)).statevector(),
    )


@pytest.mark.parametrize("basis", range(4))
def test_all_nonzero_data_columns(basis: int) -> None:
    amplitudes = tuple(float(i == basis) for i in range(4))
    assert_complete(build_program(amplitudes=amplitudes), lcu_matrix()[:, basis])


@pytest.mark.parametrize(
    "budget", ["operations", "width", "workspace", "depth", "memory"]
)
def test_actual_resource_constraints_refuse(budget: str) -> None:
    bindings: dict[str, float | list[float]] = {
        "probabilities": list(PROBABILITIES),
        "amplitudes": list(ZERO_INPUT),
    }
    with pytest.raises((HybridVerificationError, synth.SynthesisError, PlanningError)):
        if budget == "operations":
            synth.synthesize(main, classical_inputs=bindings, max_operations=1)
        elif budget == "width":
            synth.synthesize(main, classical_inputs=bindings, max_qubits=4)
        elif budget == "workspace":
            synth.synthesize(main, classical_inputs=bindings, max_workspace=0)
        elif budget == "depth":
            synth.synthesize(main, classical_inputs=bindings, max_depth=1)
        else:
            synth.synthesize(main, classical_inputs=bindings, memory_limit_bytes=16)
