from __future__ import annotations

import pytest
import torch

import flagquantum as fq

pytestmark = pytest.mark.integration


def test_plan_run_and_circuit_methods_share_resolved_options() -> None:
    circuit = fq.Circuit(3).h(0).cx(0, 1).ry(2, theta=0.2)
    options = fq.ExecutionOptions(
        mode="mps",
        precision="complex64",
        require_gradients=False,
        allow_approximate=False,
    )

    functional_plan = fq.plan(circuit, options=options)
    method_plan = circuit.plan(options=options)
    functional_result = fq.run(circuit, options=options)
    method_result = circuit.run(options=options)

    assert functional_plan.summary() == method_plan.summary()
    assert functional_result.plan.summary() == functional_plan.summary()
    assert method_result.plan.summary() == functional_plan.summary()
    torch.testing.assert_close(
        functional_result.to_statevector(), method_result.to_statevector()
    )


def test_shot_options_create_reproducible_sample_request() -> None:
    circuit = fq.Circuit(2).h(0).cx(0, 1)
    options = fq.ExecutionOptions(
        mode="statevector",
        target="samples",
        shots=12,
        seed=17,
    )

    first = fq.run(circuit, options=options)
    second = fq.run(circuit, options=options)

    assert first.samples.shape == (1, 12, 2)
    torch.testing.assert_close(first.samples, second.samples)
    assert tuple(item.kind for item in first.measurements) == ("sample",)


def test_stable_execution_rejects_unknown_and_backend_specific_keywords() -> None:
    circuit = fq.Circuit(1)

    with pytest.raises(TypeError, match="unexpected keyword"):
        fq.run(circuit, mode="mps")
    with pytest.raises(TypeError, match="unexpected keyword"):
        circuit.plan(max_bond=4)


def test_stable_execution_fails_closed_for_unavailable_backend() -> None:
    circuit = fq.Circuit(1)
    options = fq.ExecutionOptions(
        backend="unregistered-backend",
        allow_backend_fallback=False,
    )

    with pytest.raises(NotImplementedError, match="is not available"):
        fq.run(circuit, options=options)


@pytest.mark.parametrize("operation", (fq.plan, fq.run))
def test_stable_execution_rejects_program_precision_demotion(operation) -> None:
    circuit = fq.Circuit(1, dtype=torch.complex128).h(0)
    options = fq.ExecutionOptions(precision="complex64")

    with pytest.raises(ValueError, match="would demote a complex128 program"):
        operation(circuit, options=options)
