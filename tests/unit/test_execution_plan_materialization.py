"""Dense replay remains exact and rechecks each execution's sealed inputs."""

from __future__ import annotations

import json
from typing import Literal

import pytest
import torch

import flagquantum as fq
from flagquantum.runtime.execution_plan_contract import ExecutionPlanContractError

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("mode", ["statevector", "density_matrix"])
def test_dense_gate_chain_replays_against_independent_matrix_product(
    mode: Literal["statevector", "density_matrix"],
) -> None:
    indices = torch.arange(8, dtype=torch.float64)
    fourier = (
        torch.exp(2j * torch.pi * indices[:, None] * indices[None, :] / 8) / 8**0.5
    )
    phases = torch.diag(torch.exp(0.37j * indices))
    permutation = torch.eye(8, dtype=torch.complex128).roll(3, dims=0)
    circuit = fq.Circuit(3, dtype=torch.complex128)
    expected = torch.zeros(8, dtype=torch.complex128)
    expected[0] = 1
    for matrix in (fourier, phases, permutation, fourier):
        circuit.any(0, 1, 2, matrix=matrix)
        expected = matrix @ expected
    plan = fq.plan(circuit, options=fq.ExecutionOptions(mode=mode))
    restored = fq.ExecutionPlan.from_json(plan.to_json())
    if mode == "density_matrix":
        expected = torch.outer(expected, expected.conj())

    for accepted in (plan, restored):
        result = fq.run(accepted)
        assert result.plan is accepted
        assert result.state is not None
        torch.testing.assert_close(result.state[0], expected, atol=1e-12, rtol=0)
        assert result.state.device.type == "cpu"
        if mode == "statevector":
            assert result.runtime["distribution_semantics"] == "single_device_fast_path"


def test_reusing_a_plan_rejects_a_changed_seal_before_numerical_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = fq.plan(fq.Circuit(1).h(0))
    fq.run(plan)
    payload = json.loads(plan.to_json())
    payload["program"]["instructions"][0]["opcode"] = "x"
    # Frozen records can still be corrupted by deserializers or external code.
    object.__setattr__(plan, "_contract_payload_json", json.dumps(payload))

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("tampered replay reached numerical execution")

    monkeypatch.setattr("flagquantum.runtime.execution.run_native", forbidden)
    with pytest.raises(ExecutionPlanContractError) as captured:
        fq.run(plan)
    assert captured.value.reason_code == "program_fingerprint_mismatch"


def test_reusing_a_plan_checks_the_live_environment_again(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = fq.plan(fq.Circuit(1).h(0))
    fq.run(plan)

    class ChangedEnvironment:
        effective_world_size = 2

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("incompatible replay reached numerical execution")

    monkeypatch.setattr("flagquantum.runtime.execution.run_native", forbidden)
    monkeypatch.setattr(
        "flagquantum.runtime.distributed.backend_policy.resolve_distributed_backend_policy",
        ChangedEnvironment,
    )
    with pytest.raises(ExecutionPlanContractError) as captured:
        fq.run(plan)
    assert captured.value.reason_code == "world_size_mismatch"
