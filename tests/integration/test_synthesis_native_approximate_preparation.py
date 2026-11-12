"""The native example retains every signed source amplitude and error budget."""

import ast
import json
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from examples.synthesis.compare_states import compare_states
from examples.synthesis.tutorials.technology_demonstrations.approximated_state_preparation.comparisons.circuit import (
    reference_circuit,
)
from examples.synthesis.tutorials.technology_demonstrations.approximated_state_preparation.main import (
    AMPLITUDES,
    build_program,
)
from flagquantum import synth
from flagquantum.compiler import ResourceEstimate

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[2]
SOURCE = (
    ROOT
    / "tests/fixtures/classiq_qmod/corpus/original/tutorials/technology_demonstrations/approximated_state_preparation/approximated_state_preparation.qmod"
)


def test_native_and_original_retain_the_complete_signed_vector() -> None:
    source = SOURCE.read_text()
    start = source.index("[", source.index("prepare_amplitudes("))
    values = ast.literal_eval(source[start : source.index("]", start) + 1])
    assert tuple(values) == AMPLITUDES
    assert len(values) == 64
    assert any(value < 0 for value in values)
    expected = torch.tensor(values, dtype=torch.complex128)
    expected /= torch.linalg.vector_norm(expected)
    for program in (build_program(), synth.synthesize(SOURCE)):
        compare_states(program, reference_circuit(), ("out",))
        torch.testing.assert_close(
            fq.run(program).statevector().flatten(), expected, atol=1e-12, rtol=0
        )
        assert program.report["workspace_qubits"] == 0
        resources = program.report["resources"]
        assert isinstance(resources, ResourceEstimate)
        assert (
            resources.n_qubits,
            sum(resources.operation_counts.values()),
            resources.depth,
        ) == (6, 125, 120)
        probability_error = torch.linalg.vector_norm(
            fq.run(program).statevector().flatten().abs().square()
            - expected.abs().square()
        )
        assert (
            float(probability_error) < 1e-12
        )  # Also satisfies the original 0.3 bound.


def test_notebook_approximation_workflow_requires_separate_acceptance() -> None:
    inventory = json.loads(
        (ROOT / "tests/fixtures/classiq_qmod/native_library_inventory.json").read_text()
    )
    group = next(
        g
        for g in inventory["groups"]
        if g["id"].endswith(
            "/approximated_state_preparation/approximated_state_preparation"
        )
    )
    assert group["native_program"]
    assert group["workflow_status"] == "validated"
    evidence = group["workflow_validation"]
    assert (
        evidence["acceptance_test"]
        == "tests/integration/test_synthesis_native_amplitude_sweep.py"
    )
    assert evidence["quantum_program_acceptance_test"] == str(
        Path(__file__).relative_to(ROOT)
    )
    assert evidence["acceptance_test"] != evidence["quantum_program_acceptance_test"]
    assert (ROOT / evidence["acceptance_test"]).is_file()
    assert evidence["reviewed_code_cells"]
    provenance = json.loads(
        (ROOT / group["native_program"]["path"])
        .with_name("provenance.json")
        .read_text()
    )
    assert evidence == provenance
