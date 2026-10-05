from __future__ import annotations

import importlib
import json
import os
import platform
import stat
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

pytestmark = pytest.mark.integration
SOURCE_CONFORMANCE = "FLAGQUANTUM_TEST_KAIWU_SOURCE"


def _require_plugin_source() -> Path:
    if os.environ.get(SOURCE_CONFORMANCE) != "1":
        pytest.skip(f"set {SOURCE_CONFORMANCE}=1 through the pinned source runner")
    module = importlib.import_module("kaiwu.torch_plugin")
    return Path(str(module.__file__)).resolve().parents[3]


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
        "requested_samples": 10,
        "remote_call_budget": 64,
        "precision_policy": {
            "name": "explicit-int8",
            "target_min": -127,
            "target_max": 127,
        },
    }


def test_injected_transport_cannot_pass_live_system_acceptance() -> None:
    plugin_root = _require_plugin_source()
    client = _IdentityClient()

    record = run_live_system_probe(
        client=client,
        config=_config(),
        config_sha256="c" * 64,
        execution_host="jp-a800-171",
        observed_hostname="test-hostname",
        source_revision="a" * 40,
        plugin_revision="b" * 40,
        source_preflight_sha256="d" * 64,
        transfer_manifest_sha256="e" * 64,
        environment_lock_sha256="f" * 64,
        sdk_version="1.3.1",
        device=torch.device("cpu"),
        observed_gpu="test CPU",
        project_no="CPQC-test",
        task_prefix="system-test",
        requested_samples=10,
        timeout=1.0,
        poll_interval=0.01,
        real_provider_transport=False,
        plugin_root=plugin_root,
    )

    assert record["run_completed"] is True
    assert record["provider_identity_complete"] is True
    assert record["transport"] == "injected_test"
    assert record["qboson_hardware_used"] is False
    assert record["real_provider_evidence"] is False
    assert record["source_preflight_sha256"] == "d" * 64
    assert record["transfer_manifest_sha256"] == "e" * 64
    assert record["environment_lock_sha256"] == "f" * 64
    assert record["acceptance"] == {"system": "fail", "application": "not_run"}
    assert record["fallback_occurred"] is False
    assert record["retrieval_resubmitted"] is False
    assert 0 < record["remote_call_count"] <= record["remote_call_budget"]
    assert client.submissions == record["remote_call_count"]
    precision = record["precision_policy"]
    assert record["precision_evidence_complete"] is True
    assert precision["matrix_count"] >= record["remote_call_count"]
    assert 0 < precision["scale_factor_min"] <= precision["scale_factor_max"]
    assert 0 <= precision["mean_of_matrix_mean_abs_error"] <= precision["max_abs_error"]
    transfers = record["transfer_accounting"]
    assert transfers["matrix_origin_device"] == "cpu"
    assert transfers["returned_sample_target_device"] == "cpu"
    assert (
        sum(
            boundary["cache_hit"] is False
            for boundary in transfers["sampler_boundaries"]
        )
        == record["remote_call_count"]
    )


@pytest.mark.parametrize(
    ("credential", "payload"),
    (
        ("sdk-code-secret", {"failure": "vendor echoed sdk-code-secret"}),
        (
            'sdk"code\\secret\nline',
            {"failure": {"nested": 'vendor sdk"code\\secret\nline echoed'}},
        ),
        ("user-id-secret", {"metadata-user-id-secret": "present in a key"}),
    ),
)
def test_private_writer_rejects_credentials_before_creating_file(
    tmp_path: Path, credential: str, payload: dict[str, object]
) -> None:
    path = tmp_path / "record.json"

    with pytest.raises(RuntimeError, match="credential"):
        _write_private_redacted_json(
            path,
            payload,
            forbidden_values=("unrelated-secret", credential),
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


def test_private_writer_syncs_file_and_parent_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    synced_types: list[int] = []
    real_fsync = os.fsync

    def record_fsync(descriptor: int) -> None:
        synced_types.append(os.fstat(descriptor).st_mode)
        real_fsync(descriptor)

    monkeypatch.setattr("examples.qdiffusion_kaiwu.private_io.os.fsync", record_fsync)

    _write_private_redacted_json(
        tmp_path / "durable.json",
        {"secrets_redacted": True},
        forbidden_values=(),
    )

    assert any(stat.S_ISREG(mode) for mode in synced_types)
    assert any(stat.S_ISDIR(mode) for mode in synced_types)


@pytest.mark.parametrize("unsafe_kind", ("missing", "public", "symlink"))
def test_private_writer_rejects_unsafe_parent(tmp_path: Path, unsafe_kind: str) -> None:
    private_parent = tmp_path / "private"
    private_parent.mkdir(mode=0o700)
    if unsafe_kind == "missing":
        path = tmp_path / "missing" / "record.json"
    elif unsafe_kind == "public":
        private_parent.chmod(0o755)
        path = private_parent / "record.json"
    else:
        linked_parent = tmp_path / "linked"
        linked_parent.symlink_to(private_parent, target_is_directory=True)
        path = linked_parent / "record.json"

    with pytest.raises(ValueError, match="existing private, non-symlink"):
        _write_private_redacted_json(
            path, {"secrets_redacted": True}, forbidden_values=()
        )

    assert not path.exists()


def test_live_system_validates_source_preflight_before_credentials() -> None:
    source = (
        Path(__file__).parents[3]
        / "examples"
        / "qdiffusion_kaiwu"
        / "qdiffusion_system_live.py"
    ).read_text(encoding="utf-8")

    assert source.index("load_source_preflight(") < source.index(
        "resolve_kaiwu_credentials()"
    )
    assert source.index("_load_pinned_qdiffusion_api(arguments.plugin_root)") < (
        source.index("resolve_kaiwu_credentials()")
    )
    assert source.index("verify_frozen_environment_lock(") < source.index(
        "resolve_kaiwu_credentials()"
    )
    assert 'parser.add_argument("--plugin-root"' in source
    assert "source_root=Path(__file__).resolve().parents[2]" in source
    assert "plugin_root=arguments.plugin_root" in source
