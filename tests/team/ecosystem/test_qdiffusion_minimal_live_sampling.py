from __future__ import annotations

import sys
from pathlib import Path

import pytest

from examples.qdiffusion_kaiwu.minimal_live_sampling import main, run_live_sampling
from flagquantum.remote.kaiwu import (
    KaiwuTaskReceipt,
    KaiwuTaskResult,
    new_receipt,
)
from flagquantum.remote.kaiwu.contracts import FrozenIsingMatrix, KaiwuTaskMode

pytestmark = pytest.mark.unit


class _CompletedClient:
    def __init__(self) -> None:
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
        return new_receipt(
            task_name=task_name,
            matrix=matrix,
            mode=mode,
            requested_samples=requested_samples,
            project_no=project_no,
            provider_task_id="provider-task",
            provider_target="SPQC-test",
        )

    def query_status(self, receipt: KaiwuTaskReceipt, matrix: FrozenIsingMatrix) -> str:
        del receipt, matrix
        return "Completed"

    def fetch_result(
        self, receipt: KaiwuTaskReceipt, matrix: FrozenIsingMatrix
    ) -> KaiwuTaskResult:
        sample = (1, -1, 1)
        energy = -sum(
            sample[row] * matrix[row][column] * sample[column]
            for row in range(len(matrix))
            for column in range(len(matrix))
        )
        return KaiwuTaskResult(
            receipt=receipt,
            samples=tuple(sample for _ in range(receipt.requested_samples)),
            energies=tuple(energy for _ in range(receipt.requested_samples)),
            raw_status="Completed",
            metadata={"fallback_occurred": False},
        )


def test_minimal_live_sampling_uses_flagquantum_boundaries(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    tmp_path.chmod(0o700)
    receipt = tmp_path / "receipt.json"
    client = _CompletedClient()

    sampler = run_live_sampling(
        client=client,
        receipt_output=receipt,
        task_name="minimal-live",
    )

    output = capsys.readouterr().out
    assert "FlagQuantum -> Kaiwu -> QBoson sampling completed" in output
    assert "assignments: 10" in output
    assert "first binary assignment: [1, 0]" in output
    assert "remote calls: 1" in output
    assert "fallback: False" in output
    assert client.submissions == 1
    assert sampler.remote_call_count == 1
    assert receipt.is_file()
    assert receipt.stat().st_mode & 0o077 == 0


def test_minimal_live_command_requires_exact_cost_acknowledgement(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "minimal_live_sampling",
            "--checkpoint-dir",
            str(tmp_path),
            "--receipt-output",
            str(tmp_path / "receipt.json"),
            "--acknowledge-provider-cost",
            "NO",
        ],
    )

    with pytest.raises(SystemExit) as failure:
        main()

    assert failure.value.code == 2
    assert "no provider client was created" in capsys.readouterr().err
    assert not (tmp_path / "receipt.json").exists()
