from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from examples.qdiffusion_kaiwu import qboson_live_smoke as smoke_module
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
            metadata={
                "fallback_occurred": False,
                "provider_result_schema": {
                    "available": False,
                    "reason": "test_client",
                },
            },
        )


def test_live_smoke_runs_both_modes_without_overclaiming() -> None:
    client = _CompletedClient(expose_provider_identity=False)

    record = run_live_smoke(
        client=client,
        task_prefix="smoke",
        project_no="CPQC-test",
        timeout=1.0,
        poll_interval=0.01,
        environment_lock_sha256="a" * 64,
        sdk_approval_sha256="f" * 64,
    )

    assert client.submissions == 2
    assert [task["task_mode"] for task in record["tasks"]] == [
        "optimization",
        "sampling",
    ]
    assert record["live_provider_smoke_passed"] is True
    assert record["provider_identity_complete"] is False
    assert record["hardware_acceptance"] is False
    assert record["transport"] == "injected_test"
    assert record["real_provider_evidence"] is False
    assert record["qboson_hardware_used"] is False
    assert record["fallback_occurred"] is False
    assert record["environment_lock_sha256"] == "a" * 64
    assert record["sdk_approval_sha256"] == "f" * 64
    assert all(
        task["provider_result_schema"] == {"available": False, "reason": "test_client"}
        for task in record["tasks"]
    )


def test_injected_live_smoke_cannot_claim_hardware_with_complete_identity() -> None:
    record = run_live_smoke(
        client=_CompletedClient(expose_provider_identity=True),
        task_prefix="smoke",
        project_no="CPQC-test",
        timeout=1.0,
        poll_interval=0.01,
        environment_lock_sha256="b" * 64,
        sdk_approval_sha256="f" * 64,
    )

    assert record["provider_identity_complete"] is True
    assert record["live_provider_smoke_passed"] is True
    assert record["hardware_acceptance"] is False


def test_live_smoke_requires_exact_sdk_client_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(smoke_module, "KaiwuSDKClient", _CompletedClient)
    record = run_live_smoke(
        client=_CompletedClient(expose_provider_identity=True),
        task_prefix="smoke",
        project_no="CPQC-test",
        timeout=1.0,
        poll_interval=0.01,
        environment_lock_sha256="c" * 64,
        sdk_approval_sha256="f" * 64,
    )

    assert record["transport"] == "kaiwu_cim"
    assert record["real_provider_evidence"] is True
    assert record["qboson_hardware_used"] is True
    assert record["hardware_acceptance"] is True


def test_live_smoke_rejects_invalid_environment_lock_digest() -> None:
    with pytest.raises(ValueError, match="environment_lock_sha256"):
        run_live_smoke(
            client=_CompletedClient(expose_provider_identity=True),
            task_prefix="smoke",
            project_no="CPQC-test",
            timeout=1.0,
            poll_interval=0.01,
            environment_lock_sha256="not-a-digest",
            sdk_approval_sha256="f" * 64,
        )


def test_live_smoke_rejects_invalid_sdk_approval_digest() -> None:
    with pytest.raises(ValueError, match="sdk_approval_sha256"):
        run_live_smoke(
            client=_CompletedClient(expose_provider_identity=True),
            task_prefix="smoke",
            project_no="CPQC-test",
            timeout=1.0,
            poll_interval=0.01,
            environment_lock_sha256="a" * 64,
            sdk_approval_sha256="not-a-digest",
        )


def test_live_smoke_retains_failed_task_and_stops_before_another_submission() -> None:
    class _PendingClient(_CompletedClient):
        def query_status(
            self, receipt: KaiwuTaskReceipt, matrix: FrozenIsingMatrix
        ) -> str:
            del receipt, matrix
            return "Pending"

    client = _PendingClient(expose_provider_identity=True)

    record = run_live_smoke(
        client=client,
        task_prefix="timeout-smoke",
        project_no="CPQC-test",
        timeout=0.0,
        poll_interval=0.01,
        environment_lock_sha256="d" * 64,
        sdk_approval_sha256="f" * 64,
    )

    assert client.submissions == 1
    assert len(record["tasks"]) == 1
    assert record["tasks"][0]["task_mode"] == "optimization"
    assert record["tasks"][0]["raw_status"] == "Pending"
    assert record["tasks"][0]["returned_samples"] is None
    assert record["run_completed"] is False
    assert record["failure"]["type"] == "TimeoutError"
    assert record["live_provider_smoke_passed"] is False
    assert record["provider_identity_complete"] is False
    assert record["qboson_hardware_used"] is False
    assert record["hardware_acceptance"] is False


def test_live_smoke_converts_keyboard_interrupt_to_failed_attempt_record() -> None:
    class _InterruptedClient(_CompletedClient):
        def query_status(
            self, receipt: KaiwuTaskReceipt, matrix: FrozenIsingMatrix
        ) -> str:
            del receipt, matrix
            raise KeyboardInterrupt

    client = _InterruptedClient(expose_provider_identity=True)

    record = run_live_smoke(
        client=client,
        task_prefix="interrupted-smoke",
        project_no="CPQC-test",
        timeout=1.0,
        poll_interval=0.01,
        environment_lock_sha256="e" * 64,
        sdk_approval_sha256="f" * 64,
    )

    assert client.submissions == 1
    assert len(record["tasks"]) == 1
    assert record["tasks"][0]["raw_status"] is None
    assert record["failure"] == {"type": "KeyboardInterrupt", "message": ""}
    assert record["run_completed"] is False
    assert record["hardware_acceptance"] is False


def test_private_record_is_exclusive_and_mode_0600(tmp_path: Path) -> None:
    path = tmp_path / "smoke.json"
    payload = {"secret": "not-a-real-credential"}

    _write_private_json(path, payload)

    assert json.loads(path.read_text()) == payload
    assert path.stat().st_mode & 0o777 == 0o600
    with pytest.raises(FileExistsError):
        _write_private_json(path, payload)


@pytest.mark.parametrize("unsafe_kind", ("missing", "public", "symlink"))
def test_private_record_rejects_unsafe_parent(tmp_path: Path, unsafe_kind: str) -> None:
    private_parent = tmp_path / "private"
    private_parent.mkdir(mode=0o700)
    if unsafe_kind == "missing":
        path = tmp_path / "missing" / "smoke.json"
    elif unsafe_kind == "public":
        private_parent.chmod(0o755)
        path = private_parent / "smoke.json"
    else:
        linked_parent = tmp_path / "linked"
        linked_parent.symlink_to(private_parent, target_is_directory=True)
        path = linked_parent / "smoke.json"

    with pytest.raises(ValueError, match="existing private, non-symlink"):
        _write_private_json(path, {"secrets_redacted": True})

    assert not path.exists()


@pytest.mark.parametrize(
    ("credential", "payload"),
    (
        (
            "sdk-code-secret",
            {"provider_result_schema": {"fields": ["sdk-code-secret"]}},
        ),
        (
            'sdk"code\\secret\nline',
            {"nested": [{"failure": 'sdk"code\\secret\nline'}]},
        ),
        ("user-id-secret", {"provider-user-id-secret": True}),
    ),
)
def test_private_record_rejects_credentials_before_file_creation(
    tmp_path: Path, credential: str, payload: dict[str, object]
) -> None:
    path = tmp_path / "smoke.json"

    with pytest.raises(RuntimeError, match="credentials"):
        _write_private_json(
            path,
            payload,
            forbidden_values=("unrelated-secret", credential),
        )

    assert not path.exists()


def test_live_smoke_verifies_environment_before_client_initialization() -> None:
    source = (
        Path(__file__).parents[3]
        / "examples"
        / "qdiffusion_kaiwu"
        / "qboson_live_smoke.py"
    ).read_text(encoding="utf-8")

    assert "--environment-lock" in source
    assert "--sdk-approval" in source
    assert source.index("load_sdk_approval(") < source.index(
        "resolve_kaiwu_credentials()"
    )
    assert source.index("verify_approved_kaiwu_distribution(") < source.index(
        "resolve_kaiwu_credentials()"
    )
    assert source.index("verify_environment_lock(") < source.index(
        "resolve_kaiwu_credentials()"
    )
    assert source.index("validate_private_json_output_path(arguments.output)") < (
        source.index("resolve_kaiwu_credentials()")
    )
    assert source.index("validate_private_directory(") < source.index(
        "resolve_kaiwu_credentials()"
    )
    assert source.index("resolve_kaiwu_credentials()") < source.index(
        "client = KaiwuSDKClient("
    )
    assert source.index('os.environ.pop("QBOSON_USER_ID", None)') < source.index(
        "client = KaiwuSDKClient("
    )
    assert source.index('os.environ.pop("QBOSON_SDK_CODE", None)') < source.index(
        "client = KaiwuSDKClient("
    )
    assert "type(client) is KaiwuSDKClient" in source
    assert 'if not payload["hardware_acceptance"]:' in source
    assert "raise SystemExit(1)" in source


def test_live_smoke_cli_rejects_existing_output_before_credentials(
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

    monkeypatch.setattr(smoke_module, "resolve_kaiwu_credentials", resolve)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "qboson_live_smoke",
            "--checkpoint-dir",
            str(tmp_path),
            "--environment-lock",
            str(tmp_path / "lock.json"),
            "--sdk-approval",
            str(tmp_path / "approval.json"),
            "--output",
            str(output),
            "--project-no",
            "CPQC-test",
            "--task-prefix",
            "smoke",
            "--acknowledge-provider-cost",
            smoke_module.ACKNOWLEDGEMENT,
        ],
    )

    with pytest.raises(FileExistsError, match="already exists"):
        smoke_module.main()

    assert credential_resolution_attempted is False
    assert output.read_text(encoding="utf-8") == "preserve"


def test_live_smoke_cli_rejects_public_checkpoint_before_credentials(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "public-checkpoint"
    checkpoint.mkdir(mode=0o755)
    credential_resolution_attempted = False

    def resolve() -> tuple[str, str]:
        nonlocal credential_resolution_attempted
        credential_resolution_attempted = True
        raise AssertionError("credentials must not be resolved")

    monkeypatch.setattr(smoke_module, "resolve_kaiwu_credentials", resolve)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "qboson_live_smoke",
            "--checkpoint-dir",
            str(checkpoint),
            "--environment-lock",
            str(tmp_path / "lock.json"),
            "--sdk-approval",
            str(tmp_path / "approval.json"),
            "--output",
            str(tmp_path / "smoke.json"),
            "--project-no",
            "CPQC-test",
            "--task-prefix",
            "smoke",
            "--acknowledge-provider-cost",
            smoke_module.ACKNOWLEDGEMENT,
        ],
    )

    with pytest.raises(ValueError, match="Kaiwu checkpoint directory"):
        smoke_module.main()

    assert credential_resolution_attempted is False
    assert not (tmp_path / "smoke.json").exists()


def test_live_smoke_cli_writes_diagnostic_then_exits_nonzero_when_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class _CLIClient(_CompletedClient):
        def __init__(self, **kwargs: object) -> None:
            assert "QBOSON_USER_ID" not in os.environ
            assert "QBOSON_SDK_CODE" not in os.environ
            credentials = kwargs["credentials"]
            assert isinstance(credentials, smoke_module.KaiwuCredentials)
            del kwargs
            super().__init__(expose_provider_identity=False)

    output = tmp_path / "smoke.json"
    monkeypatch.setenv("QBOSON_USER_ID", "test-user")
    monkeypatch.setenv("QBOSON_SDK_CODE", "test-sdk-code")
    monkeypatch.setattr(smoke_module, "KaiwuSDKClient", _CLIClient)
    monkeypatch.setattr(
        smoke_module,
        "verify_environment_lock",
        lambda path: (
            {
                "distributions": [
                    {
                        "name": "kaiwu",
                        "version": "1.3.1",
                        "approved_artifact_sha256": "a" * 64,
                    }
                ]
            },
            "d" * 64,
        ),
    )
    approval = {
        "schema": "flagquantum.qboson_kaiwu_sdk_approval",
        "version": "1.0",
        "distribution": "kaiwu",
        "sdk_version": "1.3.1",
        "wheel_filename": "kaiwu-1.3.1-cp310-none-manylinux1_x86_64.whl",
        "source_url": "https://pypi.org/pypi/kaiwu/1.3.1/json",
        "sha256": "a" * 64,
        "service_terms_url": (
            "https://platform.qboson.com/agreement?"
            "type=QBoson-SPQC-Platform-Users-Agreement"
        ),
        "service_terms_effective_date": "2026-07-09",
        "rights_reviewed_at": "2026-10-06T00:00:00Z",
        "approval_reference": "LEGAL-APPROVAL-1",
        "organizational_use_approved": True,
        "isolated_container_use_approved": True,
        "host_staging_approved": True,
        "adapter_distribution_approved": True,
        "sdk_redistribution_policy": "no-sdk-redistribution",
    }
    monkeypatch.setattr(
        smoke_module,
        "load_sdk_approval",
        lambda path: (approval, "e" * 64),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "qboson_live_smoke",
            "--checkpoint-dir",
            str(tmp_path),
            "--environment-lock",
            str(tmp_path / "lock.json"),
            "--sdk-approval",
            str(tmp_path / "approval.json"),
            "--output",
            str(output),
            "--project-no",
            "CPQC-test",
            "--task-prefix",
            "smoke",
            "--acknowledge-provider-cost",
            smoke_module.ACKNOWLEDGEMENT,
        ],
    )

    with pytest.raises(SystemExit) as raised:
        smoke_module.main()

    assert raised.value.code == 1
    assert (
        json.loads(output.read_text(encoding="utf-8"))["hardware_acceptance"] is False
    )
