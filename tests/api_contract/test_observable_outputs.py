"""Public observable and output requests remain mathematical and backend-neutral."""

from __future__ import annotations

import pytest
import torch

import flagquantum as fq

pytestmark = pytest.mark.unit


def test_pauli_observables_and_hamiltonian_expectation() -> None:
    circuit = fq.Circuit(2).h(0).cx(0, 1)

    outputs = (
        fq.expectation(fq.X(0) @ fq.X(1), name="xx"),
        fq.expectation(fq.Y(0) @ fq.Y(1), name="yy"),
        fq.expectation(0.5 * (fq.X(0) @ fq.X(1)) - fq.Z(0) + 2 * fq.I()),
    )
    result = fq.run(circuit, outputs=outputs)

    torch.testing.assert_close(result.expectation("xx"), torch.tensor([1.0]))
    torch.testing.assert_close(result.expectation("yy"), torch.tensor([-1.0]))
    torch.testing.assert_close(result.expectation(2), torch.tensor([2.5]))
    assert len(result.expectations) == 3


def test_hamiltonian_expectation_preserves_autograd() -> None:
    theta = torch.tensor(0.3, requires_grad=True)
    result = fq.run(
        fq.Circuit(1).ry(0, theta=theta),
        outputs=fq.expectation(2 * fq.Z(0) + fq.X(0)),
    )

    result.expectation().sum().backward()

    torch.testing.assert_close(
        theta.grad,
        torch.cos(theta) - 2 * torch.sin(theta),
    )


def test_probability_sample_and_count_outputs() -> None:
    circuit = fq.Circuit(2).h(0).cx(0, 1)

    probabilities = fq.run(circuit, outputs=fq.probabilities(wires=(1, 0)))
    samples = fq.run(circuit, outputs=fq.samples(wires=0), shots=8)
    counts = fq.run(circuit, outputs=fq.counts(), shots=16)

    torch.testing.assert_close(
        probabilities.probabilities,
        torch.tensor([[0.5, 0.0, 0.0, 0.5]]),
    )
    assert samples.samples is not None
    assert samples.samples.shape == (1, 8, 1)
    assert sum(counts.counts[0].values()) == 16


def test_pauli_basis_sampling_uses_requested_axes() -> None:
    circuit = fq.Circuit(2).h(0).h(1).s(1)

    result = fq.run(
        circuit,
        outputs=(
            fq.samples(fq.X(0) @ fq.Y(1)),
            fq.counts(fq.X(0) @ fq.Y(1)),
        ),
        shots=12,
    )

    assert result.samples is not None
    assert result.samples.shape == (1, 12, 2)
    assert result.samples.eq(0).all()
    assert result.counts == [{"00": 12}]

    with pytest.raises(ValueError, match="exactly one Pauli product"):
        fq.samples(fq.X(0) + fq.Z(0))


def test_output_validation_fails_before_execution() -> None:
    circuit = fq.Circuit(1)

    with pytest.raises(ValueError, match="requires shots"):
        fq.run(circuit, outputs=fq.samples())
    with pytest.raises(ValueError, match="disjoint wires"):
        _ = fq.X(0) @ fq.Z(0)
    with pytest.raises(ValueError, match="outside"):
        fq.run(circuit, outputs=fq.expectation(fq.X(2)))
    with pytest.raises(TypeError, match="OutputRequest"):
        fq.run(circuit, outputs=("probabilities",))


def test_output_requests_are_serialized_in_an_explicit_plan() -> None:
    circuit = fq.Circuit(2).h(0).cx(0, 1)

    plan = fq.plan(circuit, outputs=fq.expectation(fq.X(0) @ fq.X(1)))
    restored = fq.ExecutionPlan.from_json(plan.to_json())
    result = fq.run(restored)

    assert restored.program_fingerprint == plan.program_fingerprint
    torch.testing.assert_close(result.expectation(), torch.tensor([1.0]))
