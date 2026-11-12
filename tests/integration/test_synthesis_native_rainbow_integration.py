"""Full integration Rainbow original/native/Circuit joint-state acceptance."""

import hashlib
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated

import pytest
import torch

import flagquantum as fq
from examples.synthesis.applications.finance.rainbow_options.rainbow_options_integration_method.comparisons.circuit import (
    reference_circuit,
)
from examples.synthesis.applications.finance.rainbow_options.rainbow_options_integration_method.main import (
    build_program,
    main,
)
from examples.synthesis.applications.finance.rainbow_options.rainbow_options_integration_method.operations import (
    integrate,
)
from examples.synthesis.applications.finance.rainbow_options.rainbow_options_integration_method.physics import (
    EXPONENT_RATE,
    PROBABILITIES,
    classical_parameters,
)
from flagquantum import synth
from flagquantum.errors import PlanningError
from flagquantum.synth import (
    H,
    Output,
    QBit,
    QNum,
    S,
    X,
    allocate,
    hadamard_transform,
    qfunc,
)

pytestmark = pytest.mark.integration
SOURCE = (
    Path(__file__).resolve().parents[1]
    / "fixtures/classiq_qmod/corpus/original/applications/finance/rainbow_options/rainbow_options_integration_method.qmod"
)


@qfunc
def integrator_with_coherent_marker(
    angles: list[float],
    score: Output[Annotated[QNum, (5, False, 2)]],
    reference: Output[Annotated[QNum, (5, False, 2)]],
    marker: Output[QBit],
) -> None:
    allocate(score)
    allocate(reference)
    allocate(marker)
    hadamard_transform(score)
    X(marker)
    H(marker)
    S(marker)
    integrate(angles, score, reference, marker)


def test_integrator_includes_equality_and_xors_nonzero_coherent_marker() -> None:
    angles, _ = classical_parameters()
    program = synth.synthesize(
        integrator_with_coherent_marker,
        classical_inputs={"angles": angles},
        memory_limit_bytes=1024**3,
        max_qubits=25,
    )
    weights = torch.tensor(
        [math.exp(EXPONENT_RATE * value / 4) for value in range(32)],
        dtype=torch.complex128,
    )
    state = (weights / weights.sum()).sqrt()
    delta = torch.eye(32, dtype=torch.complex128)[:, 0] - state
    preparation = torch.eye(32, dtype=torch.complex128) - 2 * torch.outer(
        delta, delta.conj()
    ) / torch.vdot(delta, delta)
    permutation = torch.zeros((2048, 2048), dtype=torch.complex128)
    for score in range(32):
        for reference in range(32):
            for marker in (0, 1):
                before = 2 * (32 * score + reference) + marker
                after = 2 * (32 * score + reference) + (marker ^ (score >= reference))
                permutation[after, before] = 1
    circuit = fq.Circuit(11, dtype=torch.complex128)
    for wire in range(5):
        circuit.h(wire)
    circuit.x(10).h(10).s(10)
    circuit.any(*range(5, 10), matrix=preparation)
    circuit.any(*range(11), matrix=permutation)
    actual = fq.run(program).statevector().reshape(2048, -1)
    expected = fq.run(circuit).statevector().flatten()
    torch.testing.assert_close(actual[:, 0], expected, atol=1e-11, rtol=0)
    assert float(actual[:, 1:].abs().square().sum()) < 1e-22


@pytest.mark.parametrize("k", [0, 1, 2])
def test_original_native_and_independent_complete_environment(k: int) -> None:
    assert (
        hashlib.sha256(SOURCE.read_bytes()).hexdigest()
        == "698afa2388b97aac99b64b83f0d7ba74af07473754b24242d44f08cda3ea8a9c"
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
    with pytest.raises(ValueError, match="readout budget exceeded"):
        program.probabilities(
            result, registers=("indicator",), max_amplitudes=actual.numel() - 1
        )
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
    angles, lower_angle = classical_parameters()
    inputs: dict[str, float | Sequence[float]] = {
        "k": 1,
        "probabilities": list(PROBABILITIES),
        "angles": angles,
        "lower_angle": lower_angle,
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


@pytest.mark.parametrize("rate", [0, -1, float("nan"), float("inf"), 5, True, 0.01])
def test_invalid_model_parameters_fail_before_capture(rate: float) -> None:
    with pytest.raises(ValueError, match="rate|probability"):
        build_program(1, rate)
