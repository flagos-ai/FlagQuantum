from __future__ import annotations

import json
import platform
from pathlib import Path

import pytest
import torch

from examples.qdiffusion_kaiwu.qdiffusion_system_live import (
    _write_private_redacted_json,
    run_live_system_probe,
)
from flagquantum.remote.kaiwu import (
    KaiwuTaskReceipt,
    KaiwuTaskResult,
    new_receipt,
)
from flagquantum.remote.kaiwu.contracts import FrozenIsingMatrix, KaiwuTaskMode

pytest.importorskip(
    "kaiwu.torch_plugin",
    reason="Kaiwu PyTorch Plugin is an optional conformance dependency",
)
pytestmark = pytest.mark.integration


class _IdentityClient:
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
            provider_task_id=f"injected-{self.submissions}",
            provider_target="injected-target",
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
            receipt=receipt,
            samples=samples,
            energies=tuple(energy for _ in samples),
            raw_status="Completed",
            metadata={"fallback_occurred": False},
        )


def _config() -> dict[str, object]:
    return {
        "primary_host": "jp-a800-171",
        "replay_host": "jp-a800-172",
        "software": {
            "source_revision": "a" * 40,
            "kaiwu_pytorch_plugin_revision": "b" * 40,
            "python_version": platform.python_version(),
            "torch_version": str(torch.__version__),
            "kaiwu_sdk_version": "1.3.1",
        },
        "remote_call_budget": 64,
        "precision_policy": {
            "name": "explicit-int8",
            "target_min": -127,
            "target_max": 127,
        },
    }


def test_injected_transport_cannot_pass_live_system_acceptance() -> None:
    client = _IdentityClient()

    record = run_live_system_probe(
        client=client,
        config=_config(),
        config_sha256="c" * 64,
        execution_host="jp-a800-171",
        observed_hostname="test-hostname",
        source_revision="a" * 40,
        plugin_revision="b" * 40,
        sdk_version="1.3.1",
        device=torch.device("cpu"),
        observed_gpu="test CPU",
        project_no="CPQC-test",
        task_prefix="system-test",
        requested_samples=10,
        timeout=1.0,
        poll_interval=0.01,
        real_provider_transport=False,
    )

    assert record["run_completed"] is True
    assert record["provider_identity_complete"] is True
    assert record["transport"] == "injected_test"
    assert record["qboson_hardware_used"] is False
    assert record["real_provider_evidence"] is False
    assert record["acceptance"] == {"system": "fail", "application": "not_run"}
    assert record["fallback_occurred"] is False
    assert record["retrieval_resubmitted"] is False
    assert 0 < record["remote_call_count"] <= record["remote_call_budget"]
    assert client.submissions == record["remote_call_count"]


def test_private_writer_rejects_credentials_before_creating_file(
    tmp_path: Path,
) -> None:
    path = tmp_path / "record.json"

    with pytest.raises(RuntimeError, match="credential"):
        _write_private_redacted_json(
            path,
            {"failure": "vendor echoed sdk-code-secret"},
            forbidden_values=("user-id-secret", "sdk-code-secret"),
        )

    assert not path.exists()


def test_private_writer_is_exclusive_and_mode_0600(tmp_path: Path) -> None:
    path = tmp_path / "record.json"
    payload = {"secrets_redacted": True}

    _write_private_redacted_json(
        path,
        payload,
        forbidden_values=("user-id-secret", "sdk-code-secret"),
    )

    assert json.loads(path.read_text()) == payload
    assert path.stat().st_mode & 0o777 == 0o600
    with pytest.raises(FileExistsError):
        _write_private_redacted_json(
            path,
            payload,
            forbidden_values=("user-id-secret", "sdk-code-secret"),
        )
