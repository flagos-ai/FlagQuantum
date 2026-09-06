from __future__ import annotations

import copy
import json
from dataclasses import replace

import pytest
import torch

import flagquantum as fq
from flagquantum.runtime.execution_plan_contract import (
    ExecutionPlanContractError,
)

pytestmark = pytest.mark.integration


def test_cpu_statevector_golden_path_is_compiled_executed_and_observable() -> None:
    circuit = fq.Circuit(2).h(0).h(0).h(0).cx(0, 1)
    options = fq.ExecutionOptions(
        mode="statevector",
        backend="pytorch",
        device="cpu",
        precision="complex128",
        allow_backend_fallback=False,
    )

    plan = fq.plan(circuit, options=options)
    result = fq.run(plan)

    assert plan.analysis.n_instructions == 2
    assert result.plan is plan
    assert result.state is not None
    assert result.state.device.type == "cpu"
    assert result.state.dtype is torch.complex128
    assert result.native() is result.state
    torch.testing.assert_close(
        result.state,
        torch.tensor([[2**-0.5, 0.0, 0.0, 2**-0.5]], dtype=torch.complex128),
        atol=1e-12,
        rtol=0,
    )

    assert result.runtime["mode"] == "statevector"
    assert result.runtime["execution_path"] == "local_statevector"
    assert result.runtime["simulation_engine"] == "pytorch_statevector"
    assert result.runtime["platform_provider"] == "pytorch_cpu"
    assert result.runtime["device"] == "cpu"
    assert result.runtime["distribution_semantics"] == "single_device_fast_path"
    assert result.provenance["requested_device"] == "cpu"
    assert result.provenance["selected_device"] == "cpu"
    assert result.provenance["cpu_fallback_used"] is False


def test_cpu_statevector_rejects_a_tampered_plan_before_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = fq.plan(
        fq.Circuit(1).h(0),
        options=fq.ExecutionOptions(
            mode="statevector",
            backend="pytorch",
            device="cpu",
            allow_backend_fallback=False,
        ),
    )
    payload = copy.deepcopy(plan.to_dict())
    payload["decision"]["precision"] = "complex128"
    tampered = replace(plan, _contract_payload_json=json.dumps(payload))

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("invalid plan reached numerical execution")

    monkeypatch.setattr("flagquantum.runtime.execution.run_native", forbidden)

    with pytest.raises(ExecutionPlanContractError) as captured:
        fq.run(tampered)

    assert captured.value.reason_code in {
        "environment_incompatible",
        "identity_mismatch",
    }
