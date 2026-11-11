"""Custom input states must never silently become the default zero state."""

from collections.abc import Callable

import pytest
import torch

import flagquantum as fq
from flagquantum.errors import CapabilityError
from flagquantum.runtime.execution import run_native

pytestmark = pytest.mark.unit


def test_issue_1129_minimal_reproduction_fails_before_planned_execution() -> None:
    inputs = torch.tensor([0, 1], dtype=torch.complex128)
    circuit = fq.Circuit(1, inputs=inputs, dtype=torch.complex128)

    torch.testing.assert_close(circuit.state().reshape(-1), inputs)
    with pytest.raises(CapabilityError, match="custom initial states"):
        fq.run(circuit)
    with pytest.raises(CapabilityError, match="custom initial states"):
        fq.plan(circuit)
    torch.testing.assert_close(inputs, torch.tensor([0, 1], dtype=inputs.dtype))


@pytest.mark.parametrize("entangled", [False, True])
def test_direct_statevector_retains_full_input_amplitudes(entangled: bool) -> None:
    inputs = torch.tensor(
        [1j, 0, 0, 1] if entangled else [0, 0, 1j, 0],
        dtype=torch.complex128,
    )
    if entangled:
        inputs = inputs / 2**0.5
    before = inputs.clone()
    circuit = fq.Circuit(2, inputs=inputs, dtype=inputs.dtype).h(0).cx(0, 1)
    h = torch.tensor([[1, 1], [1, -1]], dtype=inputs.dtype) / 2**0.5
    cx = torch.tensor(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
        dtype=inputs.dtype,
    )
    expected = cx @ torch.kron(h, torch.eye(2, dtype=inputs.dtype)) @ inputs

    torch.testing.assert_close(circuit.state().reshape(-1), expected)
    torch.testing.assert_close(
        run_native(circuit, mode="statevector").reshape(-1), expected
    )
    for entry in (fq.plan, fq.run):
        with pytest.raises(CapabilityError, match="Circuit.state"):
            entry(circuit)
    torch.testing.assert_close(inputs, before, rtol=0, atol=0)


@pytest.mark.parametrize("entry", [fq.plan, fq.run])
@pytest.mark.parametrize(
    "options",
    [
        fq.ExecutionOptions(mode="statevector"),
        fq.ExecutionOptions(mode="density_matrix"),
        fq.ExecutionOptions(mode="mps"),
        fq.ExecutionOptions(mode="tensor_network"),
        fq.ExecutionOptions(mode="stabilizer"),
        fq.ExecutionOptions(backend="jax"),
    ],
)
def test_unsupported_initial_state_plans_fail_before_backend_selection(
    entry: Callable[..., object], options: fq.ExecutionOptions
) -> None:
    circuit = fq.Circuit(1, inputs=torch.tensor([0j, 1j]))
    with pytest.raises(CapabilityError, match="custom initial states"):
        entry(circuit, options=options)


def test_explicit_input_zero_state_also_requires_a_state_transport_contract() -> None:
    circuit = fq.Circuit(1, inputs=torch.tensor([1j, 0j]))
    with pytest.raises(CapabilityError, match="custom initial states"):
        fq.plan(circuit)


def test_default_initial_state_plans_still_round_trip_and_execute() -> None:
    circuit = fq.Circuit(2, dtype=torch.complex128).h(0).cx(0, 1)
    plan = fq.plan(circuit)
    restored = fq.ExecutionPlan.from_json(plan.to_json())
    expected = circuit.state()

    for result in (fq.run(circuit), fq.run(plan), fq.run(restored)):
        torch.testing.assert_close(result.statevector(), expected)
        assert result.plan.identity == plan.identity
