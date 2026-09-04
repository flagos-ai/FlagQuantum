from __future__ import annotations

import pytest
import torch

import flagquantum as fq

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
