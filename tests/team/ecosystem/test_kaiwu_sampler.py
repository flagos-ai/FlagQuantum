from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

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

pytestmark = pytest.mark.unit


class _CompletedClient:
    def __init__(self) -> None:
        self.submitted: list[FrozenIsingMatrix] = []

    def submit(
        self,
        matrix: FrozenIsingMatrix,
        *,
        task_name: str,
        mode: KaiwuTaskMode,
        requested_samples: int,
        project_no: str | None,
    ) -> KaiwuTaskReceipt:
        self.submitted.append(matrix)
        return new_receipt(
            task_name=task_name,
            matrix=matrix,
            mode=mode,
            requested_samples=requested_samples,
            project_no=project_no,
            provider_task_id=f"fake-{len(self.submitted)}",
            provider_target="SPQC-fake",
        )

    def query_status(self, receipt: KaiwuTaskReceipt, matrix: FrozenIsingMatrix) -> str:
        del receipt, matrix
        return "Completed"

    def fetch_result(
        self, receipt: KaiwuTaskReceipt, matrix: FrozenIsingMatrix
    ) -> KaiwuTaskResult:
        sample = (1, -1, 1)
        samples = tuple(sample for _ in range(receipt.requested_samples))
        energy = -sum(
            sample[row] * matrix[row][column] * sample[column]
            for row in range(len(matrix))
            for column in range(len(matrix))
        )
        energies = tuple(energy for _ in samples)
        return KaiwuTaskResult(
            receipt,
            samples,
            energies,
            "Completed",
            {"fallback_occurred": False},
        )


class _RecoveringClient(_CompletedClient):
    def __init__(self) -> None:
        super().__init__()
        self.status_calls = 0

    def query_status(self, receipt: KaiwuTaskReceipt, matrix: FrozenIsingMatrix) -> str:
        del receipt, matrix
        self.status_calls += 1
        return "Queued" if self.status_calls == 1 else "Completed"


def test_sampler_returns_plugin_compatible_numpy_spins() -> None:
    client = _CompletedClient()
    sampler = KaiwuSampler(
        client=client,
        task_name="qdiffusion",
        requested_samples=10,
        max_remote_calls=2,
        poll_interval=0.01,
    )

    result = sampler.solve(
        np.array([[0.0, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.0]])
    )

    assert result.shape == (10, 3)
    assert result.dtype == np.int8
    assert set(np.unique(result)) == {-1, 1}
    assert sampler.remote_call_count == 1
    assert sampler.last_result is not None
    assert sampler.last_result.metadata["fallback_occurred"] is False


def test_identical_matrix_is_deduplicated_and_returns_a_copy() -> None:
    client = _CompletedClient()
    sampler = KaiwuSampler(client=client, task_name="dedupe", max_remote_calls=1)
    matrix = [[0.0, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.0]]

    first = sampler.solve(matrix)
    first[0, 0] = -1
    second = sampler.solve(matrix)

    assert second[0, 0] == 1
    assert sampler.remote_call_count == 1
    assert len(client.submitted) == 1


def test_remote_call_budget_fails_before_second_unique_submission() -> None:
    client = _CompletedClient()
    sampler = KaiwuSampler(client=client, task_name="budget", max_remote_calls=1)
    sampler.solve([[0.0, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.0]])

    with pytest.raises(RuntimeError, match="budget exhausted"):
        sampler.solve([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])

    assert len(client.submitted) == 1


def test_retry_after_timeout_resumes_same_job_without_resubmission() -> None:
    client = _RecoveringClient()
    sampler = KaiwuSampler(
        client=client,
        task_name="resume",
        timeout=0.0,
        max_remote_calls=1,
    )
    matrix = [[0.0, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.0]]

    with pytest.raises(TimeoutError):
        sampler.solve(matrix)
    result = sampler.solve(matrix)

    assert result.shape == (10, 3)
    assert len(client.submitted) == 1
    assert sampler.remote_call_count == 1
    assert len(sampler.receipts) == 1


def test_integer_precision_is_applied_only_when_explicit() -> None:
    plain_client = _CompletedClient()
    plain = KaiwuSampler(client=plain_client, task_name="plain")
    scaled_client = _CompletedClient()
    scaled = KaiwuSampler(
        client=scaled_client,
        task_name="scaled",
        integer_target_range=(-7, 7),
    )
    matrix = [[0.0, 0.25, 0.0], [0.25, 0.0, 0.5], [0.0, 0.5, 0.0]]

    plain.solve(matrix)
    scaled.solve(matrix)

    assert plain_client.submitted[0] == (
        (0.0, 0.25, 0.0),
        (0.25, 0.0, 0.5),
        (0.0, 0.5, 0.0),
    )
    assert scaled_client.submitted[0] == (
        (0.0, 4.0, 0.0),
        (4.0, 0.0, 7.0),
        (0.0, 7.0, 0.0),
    )
    assert plain.last_precision_report is None
    assert scaled.last_precision_report is not None
    assert plain.precision_reports == ()
    assert len(scaled.precision_reports) == 1


def test_precision_reports_cover_distinct_inputs_that_share_one_remote_matrix() -> None:
    client = _CompletedClient()
    sampler = KaiwuSampler(
        client=client,
        task_name="precision-collision",
        integer_target_range=(-1, 1),
        max_remote_calls=1,
    )
    first = [[0.0, 0.4, 0.0], [0.4, 0.0, 1.0], [0.0, 1.0, 0.0]]
    second = [[0.0, 0.49, 0.0], [0.49, 0.0, 1.0], [0.0, 1.0, 0.0]]

    sampler.solve(first)
    sampler.solve(second)

    assert sampler.remote_call_count == 1
    assert len(client.submitted) == 1
    assert len(sampler.precision_reports) == 2
    assert [report.max_abs_error for report in sampler.precision_reports] == [
        pytest.approx(0.4),
        pytest.approx(0.49),
    ]


def test_result_identity_failure_is_not_replaced_by_fallback() -> None:
    class WrongIdentityClient(_CompletedClient):
        def fetch_result(
            self, receipt: KaiwuTaskReceipt, matrix: FrozenIsingMatrix
        ) -> KaiwuTaskResult:
            result = super().fetch_result(receipt, matrix)
            return replace(result, receipt=replace(receipt, task_name="wrong"))

    sampler = KaiwuSampler(client=WrongIdentityClient(), task_name="no-fallback")

    with pytest.raises(RuntimeError, match="does not match"):
        sampler.solve([[0.0, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.0]])

    assert sampler.remote_call_count == 1
    assert sampler.last_result is None
