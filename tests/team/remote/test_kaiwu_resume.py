from __future__ import annotations

from pathlib import Path

import pytest

from examples.qdiffusion_kaiwu import qboson_resume as resume_module
from examples.qdiffusion_kaiwu.qboson_live_smoke import SMOKE_TASK_FIELDS
from examples.qdiffusion_kaiwu.qboson_resume import (
    RESUME_RECORD_FIELDS,
    run_resume,
)
from flagquantum.remote.kaiwu import (
    KaiwuRemoteJob,
    KaiwuTaskReceipt,
    KaiwuTaskResult,
    new_receipt,
)
from flagquantum.remote.kaiwu.contracts import FrozenIsingMatrix, KaiwuTaskMode

pytestmark = pytest.mark.unit

MATRIX: FrozenIsingMatrix = ((0.0, 1.0), (1.0, 0.0))


class _ResumeClient:
    def __init__(self, *, pending: bool = False) -> None:
        self.pending = pending
        self.submit_calls = 0
        self.status_calls = 0
        self.result_calls = 0

    def submit(
        self,
        matrix: FrozenIsingMatrix,
        *,
        task_name: str,
        mode: KaiwuTaskMode,
        requested_samples: int,
        project_no: str | None,
    ) -> KaiwuTaskReceipt:
        del matrix, task_name, mode, requested_samples, project_no
        self.submit_calls += 1
        raise AssertionError("resume must not call the FlagQuantum submit API")

    def query_status(self, receipt: KaiwuTaskReceipt, matrix: FrozenIsingMatrix) -> str:
        del receipt, matrix
        self.status_calls += 1
        return "Pending" if self.pending else "Completed"

    def fetch_result(
        self, receipt: KaiwuTaskReceipt, matrix: FrozenIsingMatrix
    ) -> KaiwuTaskResult:
        self.result_calls += 1
        sample = (1, -1)
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
            metadata={
                "fallback_occurred": False,
                "provider_result_schema": {
                    "available": False,
                    "reason": "test_client",
                },
            },
        )


def _save_receipt(
    parent: Path,
    client: _ResumeClient,
    *,
    mode: KaiwuTaskMode = "sampling",
    project_no: str = "CPQC-test",
) -> Path:
    parent.chmod(0o700)
    receipt = new_receipt(
        task_name="retained-task",
        matrix=MATRIX,
        mode=mode,
        requested_samples=10,
        project_no=project_no,
        provider_task_id="provider-task",
        provider_target="SPQC-test",
    )
    path = parent / "receipt.json"
    KaiwuRemoteJob(receipt, MATRIX, client).save(path)
    return path


def test_resume_queries_existing_identity_without_submit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    client = _ResumeClient()
    path = _save_receipt(tmp_path, client)
    monkeypatch.setattr(resume_module, "KaiwuSDKClient", _ResumeClient)

    record = run_resume(
        client=client,
        recovery_receipt=path,
        expected_mode="sampling",
        project_no="CPQC-test",
        timeout=1.0,
        poll_interval=0.01,
        environment_lock_sha256="b" * 64,
        sdk_approval_sha256="c" * 64,
    )

    assert client.submit_calls == 0
    assert client.status_calls == 1
    assert client.result_calls == 1
    assert set(record) == RESUME_RECORD_FIELDS
    assert set(record["task"]) == SMOKE_TASK_FIELDS
    assert record["resume_completed"] is True
    assert record["provider_result_validated"] is True
    assert record["provider_identity_complete"] is True
    assert record["real_provider_evidence"] is True
    assert record["qboson_hardware_used"] is True
    assert record["flagquantum_submit_api_called"] is False
    assert record["fallback_occurred"] is False


def test_resume_retains_timeout_without_submit(tmp_path: Path) -> None:
    client = _ResumeClient(pending=True)
    path = _save_receipt(tmp_path, client)

    record = run_resume(
        client=client,
        recovery_receipt=path,
        expected_mode="sampling",
        project_no="CPQC-test",
        timeout=0.0,
        poll_interval=0.01,
        environment_lock_sha256="e" * 64,
        sdk_approval_sha256="f" * 64,
    )

    assert client.submit_calls == 0
    assert client.status_calls == 1
    assert client.result_calls == 0
    assert record["resume_completed"] is False
    assert record["provider_result_validated"] is False
    assert record["failure"] == {"type": "TimeoutError", "message": ""}
    assert record["task"]["raw_status"] == "Pending"


@pytest.mark.parametrize(
    ("expected_mode", "project_no", "message"),
    (
        ("optimization", "CPQC-test", "mode"),
        ("sampling", "CPQC-other", "project"),
    ),
)
def test_resume_rejects_identity_mismatch_before_query(
    tmp_path: Path,
    expected_mode: KaiwuTaskMode,
    project_no: str,
    message: str,
) -> None:
    client = _ResumeClient()
    path = _save_receipt(tmp_path, client)

    with pytest.raises(ValueError, match=message):
        run_resume(
            client=client,
            recovery_receipt=path,
            expected_mode=expected_mode,
            project_no=project_no,
            timeout=1.0,
            poll_interval=0.01,
            environment_lock_sha256="2" * 64,
            sdk_approval_sha256="3" * 64,
        )

    assert client.submit_calls == 0
    assert client.status_calls == 0
    assert client.result_calls == 0


def test_resume_cli_preflights_inputs_before_credentials() -> None:
    source = resume_module.__file__
    assert source is not None
    text = Path(source).read_text(encoding="utf-8")

    credential_index = text.index("resolve_kaiwu_credentials()")
    assert text.index("validate_private_json_output_path(arguments.output)") < (
        credential_index
    )
    assert text.index("validate_private_directory(") < credential_index
    assert text.index("read_private_bytes(") < credential_index
    assert text.index("load_sdk_approval(") < credential_index
    assert text.index("verify_environment_lock(") < credential_index
    assert text.index("verify_approved_kaiwu_distribution(") < credential_index
    assert text.index('os.environ.pop("QBOSON_USER_ID", None)') < text.index(
        "client = KaiwuSDKClient("
    )
    assert text.index('os.environ.pop("QBOSON_SDK_CODE", None)') < text.index(
        "client = KaiwuSDKClient("
    )
    assert "submit_kaiwu_task" not in text
