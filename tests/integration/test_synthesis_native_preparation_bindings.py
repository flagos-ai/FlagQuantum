"""Bound preparation vectors retain state, ownership and resource semantics."""

import ast
from typing import TypedDict

import pytest
import torch

import flagquantum as fq
from examples.synthesis.compare_states import compare_states
from examples.synthesis.functions.state_preparation.example_prepare_state.main import (
    main as literal_distribution,
)
from examples.synthesis.functions.state_preparation.prepare_state_and_amplitudes.main import (
    distribution,
    run_example,
    signed_state,
)
from flagquantum import synth
from flagquantum.compiler import ResourceEstimate
from flagquantum.compiler._construction.frontend_python import _QuantumFunction
from flagquantum.compiler._hybrid.verifier import HybridVerificationError
from flagquantum.synth import Const, Output, QBit, QNum, allocate, prepare_state, qfunc

pytestmark = pytest.mark.integration


class ResourceLimits(TypedDict, total=False):
    max_qubits: int
    max_operations: int
    memory_limit_bytes: int


def reflection(values: list[float]) -> fq.Circuit:
    target = torch.tensor(values, dtype=torch.complex128)
    identity = torch.eye(len(values), dtype=torch.complex128)
    delta = identity[:, 0] - target
    matrix = (
        identity
        if delta.norm() == 0
        else identity - (2 * torch.outer(delta, delta.conj()) / delta.norm().square())
    )
    width = len(values).bit_length() - 1
    return fq.Circuit(width, dtype=torch.complex128).any(*range(width), matrix=matrix)


@pytest.mark.parametrize(
    "probabilities",
    [[0.05, 0.11, 0.13, 0.23, 0.27, 0.12, 0.03, 0.06], [0.1, 0.2, 0.3, 0.4]],
)
def test_bound_probabilities_have_positive_absolute_amplitudes(
    probabilities: list[float],
) -> None:
    program = synth.synthesize(
        distribution, classical_inputs={"probabilities": probabilities}
    )
    expected = torch.tensor(probabilities, dtype=torch.float64).sqrt()
    compare_states(program, reflection(expected.tolist()), ("x",))
    actual = fq.run(program).statevector().flatten()
    torch.testing.assert_close(actual.real, expected, atol=1e-12, rtol=0)
    assert float((actual.abs().square() - expected.square()).norm()) <= 0.01
    assert program.report["workspace_qubits"] == 0


@pytest.mark.parametrize("amplitudes", [[-0.6, 0.8], [-0.5, 0.5, -0.5, 0.5]])
def test_bound_signed_amplitudes_preserve_absolute_phase(
    amplitudes: list[float],
) -> None:
    program = synth.synthesize(
        signed_state, classical_inputs={"amplitudes": amplitudes}
    )
    compare_states(program, reflection(amplitudes), ("x",))
    assert program.report["workspace_qubits"] == 0


def test_largest_bound_vector_and_one_hot_preparation() -> None:
    values = [0.0] * 255 + [1.0]
    program = synth.synthesize(distribution, classical_inputs={"probabilities": values})
    compare_states(program, reflection(values), ("x",))
    assert len(program.registers["x"]) == 8


@qfunc
def one_bit(probabilities: list[float], x: Output[QBit]) -> None:
    prepare_state(probabilities, 0, x)


def test_bound_vector_must_match_declared_width() -> None:
    with pytest.raises(HybridVerificationError, match="width"):
        synth.synthesize(one_bit, classical_inputs={"probabilities": [0.25] * 4})


def test_bound_vector_is_snapshotted_and_matches_literal_resources() -> None:
    values = [0.05, 0.11, 0.13, 0.23, 0.27, 0.12, 0.03, 0.06]
    bound = synth.synthesize(distribution, classical_inputs={"probabilities": values})
    literal = synth.synthesize(literal_distribution)
    values[:] = [1.0, 0.0]
    torch.testing.assert_close(
        fq.run(bound).statevector(), fq.run(literal).statevector(), atol=1e-12, rtol=0
    )
    a, b = bound.report["resources"], literal.report["resources"]
    assert isinstance(a, ResourceEstimate) and isinstance(b, ResourceEstimate)
    assert (a.n_qubits, a.n_operations, a.depth) == (
        b.n_qubits,
        b.n_operations,
        b.depth,
    )


@qfunc
def helper(probabilities: list[float], x: Output[QNum]) -> None:
    prepare_state(probabilities, 0, x)


@qfunc
def through_helper(probabilities: list[float], x: Output[QNum]) -> None:
    helper(probabilities, x)


def test_parameterized_output_helper_keeps_its_existing_refusal() -> None:
    with pytest.raises(HybridVerificationError, match="plain writable quantum ports"):
        synth.synthesize(through_helper, classical_inputs={"probabilities": [0.2, 0.8]})


@pytest.mark.parametrize(
    "values",
    [
        [],
        [1.0],
        [0.2, 0.2, 0.6],
        [0.0] * 512,
        [-0.1, 1.1],
        [0.4, 0.4],
        [True, False],
        [float("nan"), 0],
        [float("inf"), 0],
    ],
)
def test_invalid_bound_probability_vectors_fail_before_execution(
    values: list[float],
) -> None:
    with pytest.raises(HybridVerificationError):
        synth.synthesize(distribution, classical_inputs={"probabilities": values})


def test_unnormalized_bound_amplitudes_are_not_repaired() -> None:
    with pytest.raises(HybridVerificationError):
        synth.synthesize(signed_state, classical_inputs={"amplitudes": [0.3, 0.4]})


_GLOBAL_VECTOR = [0.5, 0.5]


@qfunc
def unbound_global(x: Output[QNum]) -> None:
    prepare_state(_GLOBAL_VECTOR, 0, x)


def test_global_arrays_require_explicit_binding() -> None:
    with pytest.raises(HybridVerificationError, match="program.prepare"):
        synth.synthesize(unbound_global)


@qfunc
def initialize_twice(probabilities: list[float], x: Output[QNum]) -> None:
    prepare_state(probabilities, 0, x)
    prepare_state(probabilities, 0, x)


@qfunc
def overwrite_const(probabilities: list[float], x: Const[QNum]) -> None:
    prepare_state(probabilities, 0, x)


@qfunc
def const_target(probabilities: list[float], x: Output[QNum]) -> None:
    allocate(1, x)
    overwrite_const(probabilities, x)


@pytest.mark.parametrize("function", [initialize_twice, const_target])
def test_bound_preparation_preserves_owned_storage_checks(
    function: _QuantumFunction,
) -> None:
    with pytest.raises(HybridVerificationError):
        synth.synthesize(function, classical_inputs={"probabilities": [0.5, 0.5]})


@pytest.mark.parametrize(
    "limits", [{"max_qubits": 1}, {"max_operations": 1}, {"memory_limit_bytes": 1}]
)
def test_bound_preparation_respects_resource_limits(limits: ResourceLimits) -> None:
    with pytest.raises(ValueError):
        synth.synthesize(
            distribution,
            classical_inputs={"probabilities": [0.1, 0.2, 0.3, 0.4]},
            **limits,
        )


def test_complete_probability_and_signed_amplitude_notebook(
    capsys: pytest.CaptureFixture[str],
) -> None:
    run_example()
    lines = capsys.readouterr().out.splitlines()
    sampled = ast.literal_eval(lines[0].removeprefix("Sampled probabilities: "))
    assert len(sampled) == 8 and sum(sampled) == pytest.approx(1.0)
    assert sampled == pytest.approx(
        [0.05, 0.11, 0.13, 0.23, 0.27, 0.12, 0.03, 0.06], abs=0.04
    )
    raw = torch.tensor(
        ast.literal_eval(lines[1].removeprefix("Raw amplitudes: ")),
        dtype=torch.complex128,
    )
    expected = torch.linspace(-1, 1, 8, dtype=torch.float64)
    expected /= expected.norm()
    torch.testing.assert_close(raw.real, expected, atol=1e-12, rtol=0)
    torch.testing.assert_close(raw.imag, torch.zeros_like(expected), atol=1e-12, rtol=0)
    aligned = ast.literal_eval(lines[2].removeprefix("Amplitudes up to global phase: "))
    assert aligned == pytest.approx((-expected).tolist(), abs=1e-12)
