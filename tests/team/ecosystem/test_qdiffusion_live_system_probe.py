from __future__ import annotations

import importlib
import json
import os
import platform
import stat
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from examples.qdiffusion_kaiwu import qdiffusion_system_live as system_module
from examples.qdiffusion_kaiwu.plan_quota import SYSTEM_MAX_CALLS_PER_HOST
from examples.qdiffusion_kaiwu.qdiffusion_system_live import (
    _validate_lane,
    _validate_requested_cuda_device,
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
        "host_identities": {
            "jp-a800-171": "node-a800-171",
            "jp-a800-172": "node-a800-172",
        },
        "software": {
            "source_revision": "a" * 40,
            "flagquantum_version": "0.2.0",
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


def test_lane_rejects_flagquantum_version_drift() -> None:
    config = _config()
    config["software"]["flagquantum_version"] = "999.0.0"  # type: ignore[index]

    with pytest.raises(ValueError, match="observed flagquantum_version differs"):
        _validate_lane(
            config,
            execution_host="jp-a800-171",
            source_revision="a" * 40,
            plugin_revision="b" * 40,
            sdk_version="1.3.1",
        )


@pytest.mark.parametrize("device_name", ("cuda", "cuda:1", "cpu", "not-a-device"))
def test_live_system_requires_explicit_cuda_zero_before_provider_use(
    device_name: str,
) -> None:
    with pytest.raises(ValueError, match="requires explicit cuda:0"):
        _validate_requested_cuda_device(device_name)


def test_live_system_accepts_explicit_cuda_zero() -> None:
    assert _validate_requested_cuda_device("cuda:0") == torch.device("cuda:0")


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
    # Quantization-equivalent matrices may deduplicate differently across
    # devices; the quota planner records the reviewed upper bound, not an exact
    # expected count.
    assert 0 < record["remote_call_count"] <= SYSTEM_MAX_CALLS_PER_HOST
    assert record["remote_call_count"] <= record["remote_call_budget"]
    assert client.submissions == record["remote_call_count"]
    precision = record["precision_policy"]
    assert record["precision_evidence_complete"] is True
    assert len(record["precision_evidence"]) == precision["matrix_count"]
    assert {
        evidence["submission_matrix_sha256"]
        for evidence in record["precision_evidence"]
    } == {receipt["matrix_sha256"] for receipt in record["task_receipts"]}
    assert {
        (
            evidence["original_matrix_sha256"],
            evidence["submission_matrix_sha256"],
            evidence["source_type"],
            evidence["source_dtype"],
        )
        for evidence in record["precision_evidence"]
    } == {
        (
            boundary["original_matrix_sha256"],
            boundary["submission_matrix_sha256"],
            boundary["input_type"],
            boundary["input_dtype"],
        )
        for boundary in record["transfer_accounting"]["sampler_boundaries"]
    }
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
    non_cached = [
        boundary
        for boundary in transfers["sampler_boundaries"]
        if boundary["cache_hit"] is False
    ]
    assert [boundary["submission_matrix_sha256"] for boundary in non_cached] == [
        receipt["matrix_sha256"] for receipt in record["task_receipts"]
    ]


def test_live_system_sdk_subclass_cannot_claim_real_transport(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class _InjectedSubclass(_IdentityClient):
        pass

    def execute_once(
        device: torch.device, *, sampler: object, **kwargs: object
    ) -> dict[str, object]:
        del device, kwargs
        sampler.solve(  # type: ignore[attr-defined]
            np.asarray([[0.0, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.0]])
        )
        return {}

    monkeypatch.setattr(system_module, "KaiwuSDKClient", _IdentityClient)
    monkeypatch.setattr(system_module, "_execute_qdiffusion_slice", execute_once)
    record = run_live_system_probe(
        client=_InjectedSubclass(),
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
        task_prefix="subclass-system-test",
        requested_samples=10,
        timeout=1.0,
        poll_interval=0.01,
        real_provider_transport=True,
        plugin_root=tmp_path,
    )

    assert record["run_completed"] is True
    assert record["transport"] == "injected_test"
    assert record["pinned_sdk_client"] is False
    assert record["qboson_hardware_used"] is False
    assert record["real_provider_evidence"] is False
    assert record["acceptance"] == {"system": "fail", "application": "not_run"}


def test_live_system_producer_rejects_cpu_slice_even_with_provider_identity(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def execute_cpu_slice(
        device: torch.device, *, sampler: object, **kwargs: object
    ) -> dict[str, object]:
        del device, kwargs
        sampler.solve(  # type: ignore[attr-defined]
            np.asarray([[0.0, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.0]])
        )
        return {
            "proposal_device": "cpu",
            "energy_device": "cpu",
            "generated_device": "cpu",
            "objective": 1.0,
            "gradient_norm": 0.5,
            "parameter_delta_max": 0.1,
            "fallback_occurred": False,
            "token_constraints_passed": True,
        }

    monkeypatch.setattr(system_module, "KaiwuSDKClient", _IdentityClient)
    monkeypatch.setattr(
        system_module, "_execute_qdiffusion_slice", execute_cpu_slice
    )
    record = run_live_system_probe(
        client=_IdentityClient(),
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
        observed_gpu="NVIDIA A800-SXM4-80GB",
        project_no="CPQC-test",
        task_prefix="cpu-slice",
        requested_samples=10,
        timeout=1.0,
        poll_interval=0.01,
        real_provider_transport=True,
        plugin_root=tmp_path,
    )

    assert record["run_completed"] is True
    assert record["provider_identity_complete"] is True
    assert record["qboson_hardware_used"] is True
    assert record["acceptance"] == {"system": "fail", "application": "not_run"}


def test_live_system_converts_keyboard_interrupt_to_failed_record(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def interrupted(*args: object, **kwargs: object) -> dict[str, object]:
        del args, kwargs
        raise KeyboardInterrupt

    monkeypatch.setattr(system_module, "_execute_qdiffusion_slice", interrupted)

    record = run_live_system_probe(
        client=_IdentityClient(),
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
        task_prefix="interrupted-system",
        requested_samples=10,
        timeout=1.0,
        poll_interval=0.01,
        real_provider_transport=False,
        plugin_root=tmp_path,
    )

    assert record["run_completed"] is False
    assert record["failure"] == {"type": "KeyboardInterrupt", "message": ""}
    assert record["remote_call_count"] == 0
    assert record["qboson_hardware_used"] is False
    assert record["acceptance"] == {"system": "fail", "application": "not_run"}


def test_live_system_preserves_proven_provider_use_after_local_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def fail_after_provider_use(
        device: torch.device, *, sampler: object, **kwargs: object
    ) -> dict[str, object]:
        del device, kwargs
        sampler.solve(  # type: ignore[attr-defined]
            np.asarray([[0.0, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.0]])
        )
        raise RuntimeError("local failure after provider use")

    monkeypatch.setattr(system_module, "KaiwuSDKClient", _IdentityClient)
    monkeypatch.setattr(
        system_module, "_execute_qdiffusion_slice", fail_after_provider_use
    )

    record = run_live_system_probe(
        client=_IdentityClient(),
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
        task_prefix="failed-after-provider-use",
        requested_samples=10,
        timeout=1.0,
        poll_interval=0.01,
        real_provider_transport=True,
        plugin_root=tmp_path,
    )

    assert record["run_completed"] is False
    assert record["failure"]["type"] == "RuntimeError"
    assert record["remote_call_count"] == 1
    assert record["pinned_sdk_client"] is True
    assert record["qboson_hardware_used"] is True
    assert record["real_provider_evidence"] is True
    assert record["acceptance"] == {"system": "fail", "application": "not_run"}


def test_live_system_receipt_only_timeout_does_not_claim_provider_use(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class _PendingClient(_IdentityClient):
        def query_status(
            self, receipt: KaiwuTaskReceipt, matrix: FrozenIsingMatrix
        ) -> str:
            del receipt, matrix
            return "Pending"

    def submit_then_timeout(
        device: torch.device, *, sampler: object, **kwargs: object
    ) -> dict[str, object]:
        del device, kwargs
        sampler.solve(  # type: ignore[attr-defined]
            np.asarray([[0.0, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.0]])
        )
        return {}

    monkeypatch.setattr(system_module, "KaiwuSDKClient", _PendingClient)
    monkeypatch.setattr(
        system_module, "_execute_qdiffusion_slice", submit_then_timeout
    )
    record = run_live_system_probe(
        client=_PendingClient(),
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
        task_prefix="receipt-only-timeout",
        requested_samples=10,
        timeout=0.0,
        poll_interval=0.01,
        real_provider_transport=True,
        plugin_root=tmp_path,
    )

    assert record["run_completed"] is False
    assert record["failure"]["type"] == "TimeoutError"
    assert record["remote_call_count"] == 1
    assert record["provider_identity_complete"] is True
    assert record["returned_samples"] is None
    assert record["qboson_hardware_used"] is False
    assert record["real_provider_evidence"] is False


def test_live_system_preserves_completed_result_when_later_receipt_times_out(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class _PartialClient(_IdentityClient):
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
                provider_task_id=(
                    f"completed-{self.submissions}" if self.submissions == 1 else None
                ),
                provider_target=(
                    "injected-target" if self.submissions == 1 else None
                ),
            )

        def query_status(
            self, receipt: KaiwuTaskReceipt, matrix: FrozenIsingMatrix
        ) -> str:
            del receipt, matrix
            return "Completed" if self.submissions == 1 else "Pending"

    def complete_then_timeout(
        device: torch.device, *, sampler: object, **kwargs: object
    ) -> dict[str, object]:
        del device, kwargs
        sampler.solve(  # type: ignore[attr-defined]
            np.asarray([[0.0, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.0]])
        )
        sampler.solve(  # type: ignore[attr-defined]
            np.asarray([[0.0, 0.5, 0.5], [0.5, 0.0, 0.0], [0.5, 0.0, 0.0]])
        )
        return {}

    monkeypatch.setattr(system_module, "KaiwuSDKClient", _PartialClient)
    monkeypatch.setattr(
        system_module, "_execute_qdiffusion_slice", complete_then_timeout
    )
    record = run_live_system_probe(
        client=_PartialClient(),
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
        task_prefix="completed-then-timeout",
        requested_samples=10,
        timeout=0.0,
        poll_interval=0.01,
        real_provider_transport=True,
        plugin_root=tmp_path,
    )

    assert record["run_completed"] is False
    assert record["failure"]["type"] == "TimeoutError"
    assert record["remote_call_count"] == 2
    assert record["provider_identity_complete"] is False
    assert record["returned_samples"] == 10
    assert record["qboson_hardware_used"] is True
    assert record["real_provider_evidence"] is True


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


def test_private_writer_requires_absolute_output_path() -> None:
    with pytest.raises(ValueError, match="output path must be absolute"):
        _write_private_redacted_json(
            Path("relative-evidence.json"),
            {"secrets_redacted": True},
            forbidden_values=(),
        )


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
    assert source.index("read_private_bytes(") < source.index(
        "resolve_kaiwu_credentials()"
    )
    assert source.index("verify_approved_kaiwu_distribution(") < source.index(
        "resolve_kaiwu_credentials()"
    )
    assert source.index("load_provider_resources(") < source.index(
        "resolve_kaiwu_credentials()"
    )
    assert source.index("assess_provider_budget(") < source.index(
        "resolve_kaiwu_credentials()"
    )
    assert source.index("validate_private_json_output_path(arguments.output)") < (
        source.index("resolve_kaiwu_credentials()")
    )
    assert source.index("validate_private_directory(") < source.index(
        "resolve_kaiwu_credentials()"
    )
    assert source.index('os.environ.pop("QBOSON_USER_ID", None)') < source.index(
        "client = KaiwuSDKClient("
    )
    assert source.index('os.environ.pop("QBOSON_SDK_CODE", None)') < source.index(
        "client = KaiwuSDKClient("
    )
    assert 'parser.add_argument("--plugin-root"' in source
    assert "source_root=Path(__file__).resolve().parents[2]" in source
    assert "plugin_root=arguments.plugin_root" in source


def test_live_system_cli_rejects_existing_output_before_credentials(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output = tmp_path / "existing.json"
    output.write_text("preserve", encoding="utf-8")
    credential_resolution_attempted = False

    def resolve() -> tuple[str, str]:
        nonlocal credential_resolution_attempted
        credential_resolution_attempted = True
        raise AssertionError("credentials must not be resolved")

    monkeypatch.setattr(system_module, "resolve_kaiwu_credentials", resolve)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "qdiffusion_system_live",
            "--config",
            str(tmp_path / "config.json"),
            "--checkpoint-dir",
            str(tmp_path),
            "--output",
            str(output),
            "--execution-host",
            "jp-a800-171",
            "--expected-hostname",
            "test-host",
            "--source-revision",
            "0" * 40,
            "--plugin-revision",
            "1" * 40,
            "--plugin-root",
            str(tmp_path),
            "--source-preflight",
            str(tmp_path / "source-preflight.json"),
            "--environment-lock",
            str(tmp_path / "environment-lock.json"),
            "--provider-resources",
            str(tmp_path / "provider-resources.json"),
            "--project-no",
            "CPQC-test",
            "--task-prefix",
            "system",
            "--acknowledge-provider-cost",
            system_module.ACKNOWLEDGEMENT,
        ],
    )

    with pytest.raises(FileExistsError, match="already exists"):
        system_module.main()

    assert credential_resolution_attempted is False
    assert output.read_text(encoding="utf-8") == "preserve"
