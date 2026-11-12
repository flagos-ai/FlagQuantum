"""Retained signed-amplitude preparation uses the existing native pipeline."""

from pathlib import Path
from typing import Literal

import pytest
import torch

import flagquantum as fq
from examples.synthesis.compare_states import compare_states
from examples.synthesis.tutorials.technology_demonstrations.auxiliary_managment.auxiliary_management.comparisons.circuit import (
    reference_circuit,
)
from examples.synthesis.tutorials.technology_demonstrations.auxiliary_managment.auxiliary_management.main import (
    AMPLITUDES,
    build_program,
    main,
)
from flagquantum import synth
from flagquantum.compiler import estimate_resources
from flagquantum.errors import PlanningError

pytestmark = pytest.mark.integration
SOURCE = (
    Path(__file__).resolve().parents[1]
    / "fixtures/classiq_qmod/corpus/original/tutorials/technology_demonstrations/auxiliary_managment/auxiliary_management.qmod"
)


@pytest.mark.parametrize("objective", ["width", "depth"])
def test_all_signed_amplitudes_preserve_original_state_and_resources(
    objective: Literal["width", "depth"],
) -> None:
    native = synth.synthesize(
        main,
        classical_inputs={"amplitudes": AMPLITUDES},
        objective=objective,
        max_workspace=0,
    )
    original = synth.synthesize(SOURCE, objective=objective)
    for program in (native, original):
        compare_states(program, reference_circuit(), ("out",))
        assert program.report["workspace_qubits"] == 0
    assert native.report["resources"] == original.report["resources"]
    assert native.report["source_resources"] == original.report["source_resources"]
    actual = fq.run(native).statevector().flatten()
    expected = torch.tensor(AMPLITUDES, dtype=torch.float64)
    expected /= expected.norm()
    torch.testing.assert_close(
        actual, expected.to(torch.complex128), atol=1e-12, rtol=0
    )
    probability_error = torch.linalg.vector_norm(
        actual.abs().square() - expected.square()
    )
    assert float(probability_error) <= 0.2
    assert actual.real[1] < 0 and actual.real[0] > 0


def test_declared_budget_is_checked_on_the_preparation() -> None:
    for qubits, operations, memory in (
        (3, None, None),
        (None, 1, None),
        (None, None, 16),
    ):
        with pytest.raises((synth.SynthesisError, PlanningError)):
            synth.synthesize(
                main,
                classical_inputs={"amplitudes": AMPLITUDES},
                max_qubits=qubits,
                max_operations=operations,
                memory_limit_bytes=memory,
            )
    assert build_program().report["workspace_qubits"] == 0


def test_complete_local_notebook_preserves_seed_width_and_actual_amplitudes(
    capsys: pytest.CaptureFixture[str],
) -> None:
    np = pytest.importorskip("numpy")

    from examples.synthesis.tutorials.technology_demonstrations.auxiliary_managment.auxiliary_management.workflow import (
        build_workflow,
        run_example,
        source_amplitudes,
    )

    before = np.random.get_state()
    values = source_amplitudes()
    after = np.random.get_state()
    assert before[0] == after[0] and before[2:] == after[2:]
    np.testing.assert_array_equal(before[1], after[1])
    torch.testing.assert_close(
        torch.tensor(values, dtype=torch.float64),
        torch.tensor(AMPLITUDES, dtype=torch.float64),
        atol=1e-14,
        rtol=0,
    )
    program = build_workflow()
    compare_states(program, reference_circuit(), ("out",))
    assert estimate_resources(program.circuit).n_qubits <= 8
    assert program.report["workspace_qubits"] == 0
    state = fq.run(program).statevector().flatten()
    torch.testing.assert_close(
        state, torch.tensor(values, dtype=torch.complex128), atol=1e-12, rtol=0
    )
    run_example()
    output = capsys.readouterr().out
    assert "ResourceEstimate" in output
