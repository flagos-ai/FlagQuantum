"""Full direct Rainbow original/native/Circuit joint-state acceptance."""

import hashlib
from collections.abc import Sequence
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from examples.synthesis.applications.finance.rainbow_options.rainbow_options_direct_method.comparisons.circuit import (
    reference_circuit,
)
from examples.synthesis.applications.finance.rainbow_options.rainbow_options_direct_method.main import (
    build_program,
    main,
)
from examples.synthesis.applications.finance.rainbow_options.rainbow_options_direct_method.physics import (
    PROBABILITIES,
    loading_angles,
    low_angle,
)
from flagquantum import synth
from flagquantum.errors import PlanningError

pytestmark = pytest.mark.integration
SOURCE = (
    Path(__file__).resolve().parents[1]
    / "fixtures/classiq_qmod/corpus/original/applications/finance/rainbow_options/rainbow_options_direct_method.qmod"
)


@pytest.mark.parametrize("k", [0, 1, 2])
def test_original_native_and_independent_complete_environment(k: int) -> None:
    assert (
        hashlib.sha256(SOURCE.read_bytes()).hexdigest()
        == "6a911a20ead2fd62de750ffc3d4c4638dd0ef07d961d008a45edbc6ce00ee913"
    )
    expected = fq.run(reference_circuit(k)).statevector().flatten()
    program = build_program(k)
    assert dict(program.registers) == {"indicator": (0,)}
    result = fq.run(program)
    actual = result.statevector().reshape(1024, -1)
    torch.testing.assert_close(actual[:, 0], expected, atol=1e-10, rtol=0)
    assert float(actual[:, 1:].abs().square().sum()) < 1e-20
    original = synth.synthesize(SOURCE, classical_inputs={"k": k})
    original_state = fq.run(original).statevector().reshape(2, 32, 4, 4, -1)
    ordered = original_state.permute(0, 3, 2, 1, 4).reshape(1024, -1)
    torch.testing.assert_close(ordered[:, 0], expected, atol=1e-10, rtol=0)
    assert float(ordered[:, 1:].abs().square().sum()) < 1e-20
    assert dict(
        program.probabilities(result, registers=("indicator",))
    ) == pytest.approx(
        {(q,): float(expected.reshape(2, -1)[q].abs().square().sum()) for q in range(2)}
    )
    reduced = actual.reshape(2, -1)
    target = expected.reshape(2, -1)
    torch.testing.assert_close(
        reduced @ reduced.mH, target @ target.mH, atol=1e-10, rtol=0
    )


@pytest.mark.parametrize("rate", [0.37, 0.51])
def test_two_nonzero_model_parameters_preserve_absolute_complete_state(
    rate: float,
) -> None:
    program = build_program(1, rate)
    actual = fq.run(program).statevector().reshape(1024, -1)
    expected = fq.run(reference_circuit(1, rate)).statevector().flatten()
    torch.testing.assert_close(actual[:, 0], expected, atol=1e-10, rtol=0)
    assert float(actual[:, 1:].abs().square().sum()) < 1e-20


@pytest.mark.parametrize("constraint", ["qubits", "workspace", "operations", "memory"])
def test_full_model_refuses_insufficient_actual_budgets(constraint: str) -> None:
    inputs: dict[str, float | Sequence[float]] = {
        "k": 1,
        "probabilities": list(PROBABILITIES),
        "angles": loading_angles(),
        "lower_angle": low_angle(),
    }
    with pytest.raises((synth.SynthesisError, PlanningError)):
        if constraint == "qubits":
            synth.synthesize(main, classical_inputs=inputs, max_qubits=10)
        elif constraint == "workspace":
            synth.synthesize(main, classical_inputs=inputs, max_workspace=0)
        elif constraint == "operations":
            synth.synthesize(main, classical_inputs=inputs, max_operations=10)
        else:
            synth.synthesize(main, classical_inputs=inputs, memory_limit_bytes=1024)


@pytest.mark.parametrize("k", [-1, 5, True, 1.5])
def test_invalid_iteration_values_fail_before_capture(k: int) -> None:
    with pytest.raises(ValueError, match="integer"):
        build_program(k)
