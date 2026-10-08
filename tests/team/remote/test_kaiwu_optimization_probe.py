from __future__ import annotations

from dataclasses import replace

import pytest

from examples.qdiffusion_kaiwu import qboson_optimization_probe as probe_module
from examples.qdiffusion_kaiwu.qboson_optimization_probe import (
    EXPECTED_KAIWU_ARTIFACT_SHA256,
    run_optimization_probe,
)
from flagquantum.remote.kaiwu import (
    KaiwuTaskReceipt,
    KaiwuTaskResult,
    new_receipt,
)
from flagquantum.remote.kaiwu.contracts import FrozenIsingMatrix, KaiwuTaskMode

pytestmark = pytest.mark.unit


class _CompletedClient:
    def __init__(self) -> None:
        self.submissions: list[dict[str, object]] = []

    def submit(
        self,
        matrix: FrozenIsingMatrix,
        *,
        task_name: str,
        mode: KaiwuTaskMode,
        requested_samples: int,
        project_no: str | None,
    ) -> KaiwuTaskReceipt:
        self.submissions.append(
            {
                "task_name": task_name,
                "mode": mode,
                "requested_samples": requested_samples,
                "project_no": project_no,
            }
        )
        return new_receipt(
            task_name=task_name,
            matrix=matrix,
            mode=mode,
            requested_samples=requested_samples,
            project_no=project_no,
        )

    def query_status(self, receipt: KaiwuTaskReceipt, matrix: FrozenIsingMatrix) -> str:
        del receipt, matrix
        return "Completed"

    def fetch_result(
        self, receipt: KaiwuTaskReceipt, matrix: FrozenIsingMatrix
    ) -> KaiwuTaskResult:
        sample = (1, -1)
        energy = -sum(
            sample[row] * matrix[row][column] * sample[column]
            for row in range(len(matrix))
            for column in range(len(matrix))
        )
        return KaiwuTaskResult(
            receipt=receipt,
            samples=(sample,),
            energies=(energy,),
            raw_status="Completed",
            metadata={
                "fallback_occurred": False,
                "provider_result_schema": {
                    "available": False,
                    "reason": "test_client",
                },
            },
        )


def _run(client: _CompletedClient, *, project_no: str | None = None) -> dict:
    return run_optimization_probe(
        client=client,
        task_name="flagquantum-development-probe",
        project_no=project_no,
        timeout=1.0,
        poll_interval=0.01,
        environment_lock_sha256="a" * 64,
        provider_resources_sha256="b" * 64,
    )


def test_probe_submits_exactly_one_optimization_without_project() -> None:
    client = _CompletedClient()

    record = _run(client)

    assert client.submissions == [
        {
            "task_name": "flagquantum-development-probe",
            "mode": "optimization",
            "requested_samples": 10,
            "project_no": None,
        }
    ]
    assert record["mode"] == "optimization"
    assert record["maximum_provider_calls"] == 1
    assert record["project_no"] is None
    assert record["run_completed"] is True
    assert record["hardware_acceptance"] is False
    assert record["qdiffusion_acceptance"] is False
    assert record["real_provider_transport_invoked"] is False
    assert record["task"]["task_mode"] == "optimization"


def test_probe_passes_optional_project_without_changing_call_bound() -> None:
    client = _CompletedClient()

    record = _run(client, project_no="CPQC-test")

    assert len(client.submissions) == 1
    assert client.submissions[0]["project_no"] == "CPQC-test"
    assert record["project_no"] == "CPQC-test"


def test_probe_retains_one_pending_attempt_and_never_resubmits() -> None:
    class _PendingClient(_CompletedClient):
        def query_status(
            self, receipt: KaiwuTaskReceipt, matrix: FrozenIsingMatrix
        ) -> str:
            del receipt, matrix
            return "Pending"

    client = _PendingClient()
    record = run_optimization_probe(
        client=client,
        task_name="pending-probe",
        project_no=None,
        timeout=0.0,
        poll_interval=0.01,
        environment_lock_sha256="a" * 64,
        provider_resources_sha256="b" * 64,
    )

    assert len(client.submissions) == 1
    assert record["run_completed"] is False
    assert record["failure"]["type"] == "TimeoutError"
    assert record["task"]["task_mode"] == "optimization"
    assert record["task"]["returned_samples"] is None


def test_probe_never_uses_provider_identity_to_open_acceptance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _IdentifiedClient(_CompletedClient):
        def submit(self, *args: object, **kwargs: object) -> KaiwuTaskReceipt:
            receipt = super().submit(*args, **kwargs)  # type: ignore[arg-type]
            return replace(
                receipt,
                provider_task_id="provider-task",
                provider_target="SPQC-test",
            )

    monkeypatch.setattr(probe_module, "KaiwuSDKClient", _IdentifiedClient)
    record = _run(_IdentifiedClient())

    assert record["real_provider_transport_invoked"] is True
    assert record["hardware_acceptance"] is False
    assert record["qdiffusion_acceptance"] is False


@pytest.mark.parametrize("field", ["environment", "resources"])
def test_probe_rejects_invalid_evidence_digest(field: str) -> None:
    arguments = {
        "client": _CompletedClient(),
        "task_name": "digest-probe",
        "project_no": None,
        "timeout": 1.0,
        "poll_interval": 0.01,
        "environment_lock_sha256": "a" * 64,
        "provider_resources_sha256": "b" * 64,
    }
    arguments[
        "environment_lock_sha256"
        if field == "environment"
        else "provider_resources_sha256"
    ] = "invalid"

    with pytest.raises(ValueError, match="sha256"):
        run_optimization_probe(**arguments)  # type: ignore[arg-type]


def test_pinned_distribution_gate_rejects_wrong_artifact() -> None:
    environment = {
        "distributions": [
            {
                "name": "kaiwu",
                "version": "1.3.1",
                "approved_artifact_sha256": "0" * 64,
            }
        ]
    }

    with pytest.raises(ValueError, match="artifact"):
        probe_module._verify_pinned_kaiwu_distribution(environment)


def test_pinned_distribution_gate_accepts_exact_wheel() -> None:
    environment = {
        "distributions": [
            {
                "name": "kaiwu",
                "version": "1.3.1",
                "approved_artifact_sha256": EXPECTED_KAIWU_ARTIFACT_SHA256,
            }
        ]
    }

    probe_module._verify_pinned_kaiwu_distribution(environment)
