"""User-level Jiuding execution path without external platform access."""

import pytest
import torch

import flagquantum as fq
from flagquantum.remote.compute import _workspace_executor as executor
from flagquantum.remote.compute import jiuding
from flagquantum.remote.compute.jiuding import JiudingClient

pytestmark = pytest.mark.unit


def test_fq_run_executes_bell_measurements_through_jiuding(monkeypatch) -> None:
    monkeypatch.setenv("JIUDING_WORKSPACE", "golden-path")
    monkeypatch.setattr(jiuding, "_DEFAULT_CLIENTS", {})
    monkeypatch.setattr(JiudingClient, "start_executor", lambda self, **kwargs: None)

    def execute_locally(self, request, *, port, timeout):
        assert port == 57621
        assert timeout == 30
        return executor.execute(request, target=request["target"], device="cpu")

    monkeypatch.setattr(JiudingClient, "_executor_request", execute_locally)

    result = fq.run(
        fq.Circuit(2).h(0).cx(0, 1),
        target="jiuding:cpu",
        outputs=(
            fq.probabilities(),
            fq.expectation(fq.X(0) @ fq.X(1), name="xx"),
        ),
    )

    torch.testing.assert_close(
        result.probabilities,
        torch.tensor([[0.5, 0.0, 0.0, 0.5]]),
    )
    torch.testing.assert_close(result.expectation("xx"), torch.tensor([1.0]))
    assert result.runtime["execution_path"] == "jiuding_workspace_executor"
    assert result.provenance["requested_target"] == "jiuding:cpu"
    assert result.provenance["selected_target"] == "jiuding:cpu"
    assert result.provenance["cpu_fallback_used"] is False
