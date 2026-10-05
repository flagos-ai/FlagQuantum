from __future__ import annotations

import json
from pathlib import Path

import pytest

from examples.qdiffusion_kaiwu.qboson_live_smoke import (
    _write_private_json,
    run_live_smoke,
)
from flagquantum.remote.kaiwu import (
    KaiwuTaskReceipt,
    KaiwuTaskResult,
    new_receipt,
)
from flagquantum.remote.kaiwu.contracts import FrozenIsingMatrix, KaiwuTaskMode

pytestmark = pytest.mark.unit


class _CompletedClient:
    def __init__(self, *, expose_provider_identity: bool) -> None:
        self.expose_provider_identity = expose_provider_identity
        self.submissions = 0

    def submit(
        self,
        matrix: FrozenIsingMatrix,
        *,
        task_name: str,
        mode: KaiwuTaskMode,
        requested_samples: int,
        project_no: str | None,
    ) -> KaiwuTaskReceipt:
        self.submissions += 1
        suffix = str(self.submissions) if self.expose_provider_identity else None
        return new_receipt(
            task_name=task_name,
            matrix=matrix,
            mode=mode,
            requested_samples=requested_samples,
            project_no=project_no,
            provider_task_id=f"provider-{suffix}" if suffix else None,
            provider_target="SPQC-test" if suffix else None,
        )

    def query_status(self, receipt: KaiwuTaskReceipt, matrix: FrozenIsingMatrix) -> str:
        del receipt, matrix
        return "Completed"

    def fetch_result(
        self, receipt: KaiwuTaskReceipt, matrix: FrozenIsingMatrix
    ) -> KaiwuTaskResult:
        sample = (1, -1)
        count = receipt.requested_samples
        samples = tuple(sample for _ in range(count))
        energy = -sum(
            sample[row] * matrix[row][column] * sample[column]
            for row in range(len(matrix))
            for column in range(len(matrix))
        )
        return KaiwuTaskResult(
            receipt=receipt,
            samples=samples,
            energies=tuple(energy for _ in samples),
            raw_status="Completed",
            metadata={"fallback_occurred": False},
        )


def test_live_smoke_runs_both_modes_without_overclaiming() -> None:
    client = _CompletedClient(expose_provider_identity=False)

    record = run_live_smoke(
        client=client,
        task_prefix="smoke",
        project_no="CPQC-test",
        timeout=1.0,
        poll_interval=0.01,
    )

    assert client.submissions == 2
    assert [task["task_mode"] for task in record["tasks"]] == [
        "optimization",
        "sampling",
    ]
    assert record["live_provider_smoke_passed"] is True
    assert record["provider_identity_complete"] is False
    assert record["hardware_acceptance"] is False
    assert record["fallback_occurred"] is False


def test_live_smoke_accepts_complete_provider_identity() -> None:
    record = run_live_smoke(
        client=_CompletedClient(expose_provider_identity=True),
        task_prefix="smoke",
        project_no="CPQC-test",
        timeout=1.0,
        poll_interval=0.01,
    )

    assert record["provider_identity_complete"] is True
    assert record["hardware_acceptance"] is True


def test_private_record_is_exclusive_and_mode_0600(tmp_path: Path) -> None:
    path = tmp_path / "smoke.json"
    payload = {"secret": "not-a-real-credential"}

    _write_private_json(path, payload)

    assert json.loads(path.read_text()) == payload
    assert path.stat().st_mode & 0o777 == 0o600
    with pytest.raises(FileExistsError):
        _write_private_json(path, payload)
