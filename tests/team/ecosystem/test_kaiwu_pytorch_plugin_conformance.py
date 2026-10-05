from __future__ import annotations

import math

import pytest
import torch

from examples.qdiffusion_kaiwu.qdiffusion_system_development_probe import (
    _run_qdiffusion_slice,
)
from flagquantum.ecosystem.kaiwu import KaiwuSampler
from flagquantum.remote.kaiwu import (
    KaiwuTaskReceipt,
    KaiwuTaskResult,
    new_receipt,
)
from flagquantum.remote.kaiwu.contracts import (
    FrozenIsingMatrix,
    KaiwuTaskMode,
)

kaiwu_plugin = pytest.importorskip(
    "kaiwu.torch_plugin",
    reason="Kaiwu PyTorch Plugin is an optional conformance dependency",
)

pytestmark = pytest.mark.integration


class _ExactShapeClient:
    def __init__(self) -> None:
        self.submit_calls = 0

    def submit(
        self,
        matrix: FrozenIsingMatrix,
        *,
        task_name: str,
        mode: KaiwuTaskMode,
        requested_samples: int,
        project_no: str | None,
    ) -> KaiwuTaskReceipt:
        self.submit_calls += 1
        return new_receipt(
            task_name=task_name,
            matrix=matrix,
            mode=mode,
            requested_samples=requested_samples,
            project_no=project_no,
            provider_task_id=f"plugin-fake-{self.submit_calls}",
            provider_target="SPQC-plugin-fake",
        )

    def query_status(self, receipt: KaiwuTaskReceipt, matrix: FrozenIsingMatrix) -> str:
        del receipt, matrix
        return "Completed"

    def fetch_result(
        self, receipt: KaiwuTaskReceipt, matrix: FrozenIsingMatrix
    ) -> KaiwuTaskResult:
        sample = tuple(1 for _ in matrix)
        samples = tuple(sample for _ in range(receipt.requested_samples))
        energy = -sum(
            sample[row] * matrix[row][column] * sample[column]
            for row in range(len(matrix))
            for column in range(len(matrix))
        )
        return KaiwuTaskResult(
            receipt,
            samples,
            tuple(energy for _ in samples),
            "Completed",
            {"fallback_occurred": False},
        )


def test_sampler_runs_through_plugin_condition_sample_without_sdk_objects() -> None:
    torch.manual_seed(7)
    machine = kaiwu_plugin.BoltzmannMachine(num_nodes=4, device="cpu")
    with torch.no_grad():
        machine.linear_bias.copy_(torch.tensor([0.0, 0.5, -0.25, 0.75]))
        machine.quadratic_coef.copy_(
            torch.tensor(
                [
                    [0.0, 0.0, 1.0, -0.5],
                    [0.0, 0.0, -0.75, 0.25],
                    [1.0, -0.75, 0.0, 0.0],
                    [-0.5, 0.25, 0.0, 0.0],
                ]
            )
        )

    client = _ExactShapeClient()
    sampler = KaiwuSampler(
        client=client,
        task_name="plugin-condition",
        requested_samples=10,
        max_remote_calls=2,
        poll_interval=0.01,
    )
    visible = torch.tensor([[0.0, 0.0], [1.0, 1.0]])

    sampled = machine.condition_sample(sampler, visible)

    assert sampled.shape == (20, 4)
    torch.testing.assert_close(sampled[:10, :2], visible[0].expand(10, -1))
    torch.testing.assert_close(sampled[10:, :2], visible[1].expand(10, -1))
    assert bool(((sampled[:, 2:] == 0) | (sampled[:, 2:] == 1)).all())
    assert client.submit_calls == 2
    assert sampler.remote_call_count == 2
    assert len(sampler.receipts) == 2


def test_qdiffusion_development_slice_uses_bounded_flagquantum_sampler() -> None:
    record = _run_qdiffusion_slice(torch.device("cpu"))

    assert math.isfinite(record["objective"])
    assert record["gradient_norm"] > 0
    assert record["parameter_delta_max"] > 0
    assert record["token_constraints_passed"] is True
    assert 0 < record["remote_call_count"] <= record["remote_call_budget"]
    assert record["task_count"] == record["remote_call_count"]
    assert record["fallback_occurred"] is False
