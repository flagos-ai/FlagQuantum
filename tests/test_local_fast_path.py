import pytest
import torch

import flagquantum as fq
import flagquantum.runtime.execution as execution

pytestmark = pytest.mark.integration


def test_local_fast_path_preflight_default_program():
    report = fq.local_fast_path_preflight()
    summary = report.summary()

    assert summary["passed"] is True
    assert summary["device"] == "cpu"
    assert set(summary["results"]) == {"statevector", "mps", "tensor_network"}
    for item in summary["results"].values():
        assert item["world_size"] == 1
        assert item["distribution_semantics"] == "single_device_fast_path"
        assert item["scalability_claim_allowed"] is False
        assert item["max_abs_error_vs_statevector"] <= 1e-6


def test_local_fast_paths_do_not_resolve_distributed_policy(monkeypatch):
    def fail_policy(*args, **kwargs):
        raise AssertionError("local fast path must not resolve distributed policy")

    monkeypatch.setattr(execution, "resolve_distributed_backend_policy", fail_policy)
    circuit = fq.Circuit(3)
    circuit.h(0).rx(1, theta=0.2).cx(0, 2)

    state_result = circuit.run(mode="statevector")
    mps_result = circuit.run(mode="mps")
    tn_result = circuit.run(mode="tensor_network")
    state = state_result.to_statevector()
    mps = mps_result.to_statevector()
    tn = tn_result.to_statevector()

    assert torch.allclose(state, circuit.state(), atol=1e-6)
    assert torch.allclose(mps, state, atol=1e-6)
    assert torch.allclose(tn, state, atol=1e-6)


def test_local_fast_path_preflight_accepts_custom_modes():
    circuit = fq.Circuit(2)
    circuit.h(0).cx(0, 1)

    report = fq.local_fast_path_preflight(circuit, modes=("statevector", "mps"))

    assert report.passed is True
    assert report.modes == ("statevector", "mps")
    assert set(report.results) == {"statevector", "mps"}
