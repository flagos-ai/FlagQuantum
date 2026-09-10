from typing import Any

import pytest
import torch

import flagquantum as fq
import flagquantum.runtime.local_preflight as preflight
from flagquantum.runtime.local_preflight import local_fast_path_preflight

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("atol", [float("nan"), float("inf"), float("-inf"), -1.0])
def test_local_preflight_rejects_invalid_tolerance(atol: float) -> None:
    with pytest.raises(ValueError, match="atol must be finite and non-negative"):
        local_fast_path_preflight(atol=atol)


def test_local_preflight_accepts_zero_tolerance() -> None:
    assert local_fast_path_preflight(modes=("statevector",), atol=0.0).passed


@pytest.mark.parametrize("invalid_value", [float("nan"), float("inf")])
def test_preflight_rejects_nonfinite_backend_results(
    monkeypatch: pytest.MonkeyPatch, invalid_value: float
) -> None:
    original_run = preflight.run_native

    def faulty_backend(*args: Any, **kwargs: Any) -> Any:
        result, plan = original_run(*args, **kwargs)
        if kwargs.get("mode") == "mps":
            state = result.to_statevector()
            return torch.full_like(state, invalid_value), plan
        return result, plan

    monkeypatch.setattr(preflight, "run_native", faulty_backend)
    report = local_fast_path_preflight(modes=("mps",))
    assert not report.passed
    assert any("non-finite" in error for error in report.errors)
    with pytest.raises(RuntimeError, match="non-finite"):
        local_fast_path_preflight(modes=("mps",), raise_on_error=True)


def test_local_fast_path_preflight_default_program():
    report = local_fast_path_preflight()
    summary = report.summary()

    assert summary["passed"] is True
    assert summary["device"] == "cpu"
    assert set(summary["results"]) == {"statevector", "mps", "tensor_network"}
    for item in summary["results"].values():
        assert item["world_size"] == 1
        assert item["distribution_semantics"] == "single_device_fast_path"
        assert item["scalability_claim_allowed"] is False
        assert item["max_abs_error_vs_statevector"] <= 1e-6


def test_local_fast_paths_use_stable_options_without_distributed_setup():
    circuit = fq.Circuit(3)
    circuit.h(0).rx(1, theta=0.2).cx(0, 2)

    state_result = circuit.run(options=fq.ExecutionOptions(mode="statevector"))
    mps_result = circuit.run(options=fq.ExecutionOptions(mode="mps"))
    tn_result = circuit.run(options=fq.ExecutionOptions(mode="tensor_network"))
    state = state_result.to_statevector()
    mps = mps_result.to_statevector()
    tn = tn_result.to_statevector()

    assert torch.allclose(state, circuit.state(), atol=1e-6)
    assert torch.allclose(mps, state, atol=1e-6)
    assert torch.allclose(tn, state, atol=1e-6)


def test_local_fast_path_preflight_accepts_custom_modes():
    circuit = fq.Circuit(2)
    circuit.h(0).cx(0, 1)

    report = local_fast_path_preflight(circuit, modes=("statevector", "mps"))

    assert report.passed is True
    assert report.modes == ("statevector", "mps")
    assert set(report.results) == {"statevector", "mps"}
