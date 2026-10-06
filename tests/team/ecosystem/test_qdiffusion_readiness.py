from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from examples.qdiffusion_kaiwu import audit_readiness as readiness_module
from examples.qdiffusion_kaiwu.audit_readiness import audit_readiness

pytestmark = pytest.mark.unit


def _paths() -> dict[str, Path]:
    return {
        "dataset": Path("/private/dataset.fasta"),
        "base_checkpoint": Path("/private/checkpoint"),
        "tokenizer": Path("/private/tokenizer"),
        "evaluation_model": Path("/private/esm2.pt"),
    }


def test_readiness_reports_missing_inputs_without_credentials() -> None:
    report = audit_readiness(
        config_path=None,
        environment_lock_path=None,
        sdk_approval_path=None,
        plugin_root=None,
        primary_source_preflight=None,
        replay_source_preflight=None,
        artifact_paths=dict.fromkeys(_paths()),
        environ={},
    )

    assert report["checks"]["credentials"] == {
        "status": "missing",
        "reason": "credential_pair_absent",
    }
    assert report["checks"]["quota"]["status"] == "blocked"
    assert report["checks"]["checkpoint_directory"] == {
        "status": "missing",
        "reason": "checkpoint_directory_absent",
    }
    assert report["ready_to_start_provider_smoke"] is False
    assert report["ready_to_start_system_probe"] is False
    assert report["ready_to_start_protein_experiment"] is False
    assert report["required_stage"] == "protein-experiment"
    assert report["required_stage_ready"] is False
    assert report["credential_values_recorded"] is False


def test_readiness_never_serializes_credential_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sdk_approval = {"approved": True}
    config = {
        "software": {
            "source_revision": "a" * 40,
            "kaiwu_pytorch_plugin_revision": "b" * 40,
            "environment_lock_sha256": "c" * 64,
        },
        "kaiwu_sdk": sdk_approval,
    }
    monkeypatch.setattr(
        readiness_module,
        "_load_frozen_config",
        lambda path: (config, "d" * 64),
    )
    monkeypatch.setattr(
        readiness_module,
        "build_quota_plan",
        lambda config, digest: {"totals": {"budget_complete": True}},
    )
    monkeypatch.setattr(
        readiness_module,
        "verify_environment_lock",
        lambda path: ({"lock": True}, "c" * 64),
    )
    monkeypatch.setattr(
        readiness_module,
        "load_sdk_approval",
        lambda path: (sdk_approval, "1" * 64),
    )
    monkeypatch.setattr(
        readiness_module,
        "verify_approved_kaiwu_distribution",
        lambda environment, approval: None,
    )
    monkeypatch.setattr(
        readiness_module,
        "load_source_preflight",
        lambda *args, **kwargs: ({"manifest_sha256": "e" * 64}, "f" * 64),
    )
    monkeypatch.setattr(
        readiness_module,
        "_inspect_artifacts",
        lambda config_path, artifact_paths: ("d" * 64, {}, {}),
    )
    monkeypatch.setattr(readiness_module, "validate_private_directory", lambda *a, **k: None)
    monkeypatch.setattr(readiness_module, "_inspect_a800_cuda_zero", lambda: "A800")

    report = audit_readiness(
        config_path=Path("/private/config.json"),
        environment_lock_path=Path("/private/environment.json"),
        sdk_approval_path=Path("/private/sdk-approval.json"),
        plugin_root=Path("/private/plugin"),
        primary_source_preflight=Path("/private/171.json"),
        replay_source_preflight=Path("/private/172.json"),
        artifact_paths=_paths(),
        checkpoint_dir=Path("/private/checkpoints"),
        environ={
            "QBOSON_USER_ID": "sensitive-user",
            "QBOSON_SDK_CODE": "sensitive-code",
            "QBOSON_PROJECT_NO": "sensitive-project",
        },
        source_root=Path("/private/source"),
    )

    encoded = json.dumps(report)
    assert "sensitive-user" not in encoded
    assert "sensitive-code" not in encoded
    assert "sensitive-project" not in encoded
    assert report["ready_to_start_provider_smoke"] is True
    assert report["ready_to_start_system_probe"] is True
    assert report["ready_to_start_protein_experiment"] is True


def test_readiness_rejects_partial_credentials() -> None:
    report = audit_readiness(
        config_path=None,
        environment_lock_path=None,
        sdk_approval_path=None,
        plugin_root=None,
        primary_source_preflight=None,
        replay_source_preflight=None,
        artifact_paths=dict.fromkeys(_paths()),
        environ={"QBOSON_USER_ID": "only-one-half"},
    )

    assert report["checks"]["credentials"] == {
        "status": "fail",
        "reason": "partial_credential_pair",
    }


@pytest.mark.parametrize(
    ("environ", "check", "reason"),
    (
        (
            {"QBOSON_USER_ID": " ", "QBOSON_SDK_CODE": "code"},
            "credentials",
            "credential_pair_invalid",
        ),
        (
            {"QBOSON_USER_ID": "user", "QBOSON_SDK_CODE": "co\nde"},
            "credentials",
            "credential_pair_invalid",
        ),
        (
            {"QBOSON_PROJECT_NO": "\u200b"},
            "project",
            "project_invalid",
        ),
    ),
)
def test_readiness_rejects_invalid_private_values_without_echoing_them(
    environ: dict[str, str], check: str, reason: str
) -> None:
    report = audit_readiness(
        config_path=None,
        environment_lock_path=None,
        sdk_approval_path=None,
        plugin_root=None,
        primary_source_preflight=None,
        replay_source_preflight=None,
        artifact_paths=dict.fromkeys(_paths()),
        environ=environ,
    )

    assert report["checks"][check] == {"status": "fail", "reason": reason}
    encoded = json.dumps(report)
    assert all(value not in encoded for value in environ.values() if value.strip())


def test_readiness_rejects_invalid_checkpoint_directory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _reject(*args: object, **kwargs: object) -> None:
        raise ValueError("not private")

    monkeypatch.setattr(readiness_module, "validate_private_directory", _reject)
    report = audit_readiness(
        config_path=None,
        environment_lock_path=None,
        sdk_approval_path=None,
        plugin_root=None,
        primary_source_preflight=None,
        replay_source_preflight=None,
        artifact_paths=dict.fromkeys(_paths()),
        checkpoint_dir=Path("/private/checkpoints"),
        environ={},
    )

    assert report["checks"]["checkpoint_directory"] == {
        "status": "fail",
        "reason": "checkpoint_directory_invalid",
    }


@pytest.mark.parametrize(
    ("cuda_available", "gpu_name", "passes"),
    (
        (False, "", False),
        (True, "NVIDIA H100 80GB HBM3", False),
        (True, "NVIDIA A800-SXM4-80GB", True),
    ),
)
def test_a800_cuda_zero_inspection(
    monkeypatch: pytest.MonkeyPatch,
    cuda_available: bool,
    gpu_name: str,
    passes: bool,
) -> None:
    monkeypatch.setattr(
        readiness_module.torch.cuda, "is_available", lambda: cuda_available
    )
    monkeypatch.setattr(
        readiness_module.torch.cuda, "get_device_name", lambda device: gpu_name
    )

    if passes:
        assert readiness_module._inspect_a800_cuda_zero() == gpu_name
    else:
        with pytest.raises(RuntimeError):
            readiness_module._inspect_a800_cuda_zero()


def test_readiness_rejects_preflights_from_different_transfer_manifests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sdk_approval = {"approved": True}
    config = {
        "software": {
            "source_revision": "a" * 40,
            "kaiwu_pytorch_plugin_revision": "b" * 40,
            "environment_lock_sha256": "c" * 64,
        },
        "kaiwu_sdk": sdk_approval,
    }
    monkeypatch.setattr(
        readiness_module,
        "_load_frozen_config",
        lambda path: (config, "d" * 64),
    )
    monkeypatch.setattr(
        readiness_module,
        "build_quota_plan",
        lambda config, digest: {"totals": {"budget_complete": True}},
    )
    monkeypatch.setattr(
        readiness_module,
        "verify_environment_lock",
        lambda path: ({"lock": True}, "c" * 64),
    )
    monkeypatch.setattr(
        readiness_module,
        "load_sdk_approval",
        lambda path: (sdk_approval, "1" * 64),
    )
    monkeypatch.setattr(
        readiness_module,
        "verify_approved_kaiwu_distribution",
        lambda environment, approval: None,
    )

    def _load_preflight(*args: object, **kwargs: object) -> tuple[dict[str, str], str]:
        host = str(kwargs["execution_host"])
        digest = "e" * 64 if host == "jp-a800-171" else "f" * 64
        return {"manifest_sha256": digest}, "0" * 64

    monkeypatch.setattr(readiness_module, "load_source_preflight", _load_preflight)
    monkeypatch.setattr(
        readiness_module,
        "_inspect_artifacts",
        lambda config_path, artifact_paths: ("d" * 64, {}, {}),
    )
    monkeypatch.setattr(readiness_module, "validate_private_directory", lambda *a, **k: None)
    monkeypatch.setattr(readiness_module, "_inspect_a800_cuda_zero", lambda: "A800")

    report = audit_readiness(
        config_path=Path("/private/config.json"),
        environment_lock_path=Path("/private/environment.json"),
        sdk_approval_path=Path("/private/sdk-approval.json"),
        plugin_root=Path("/private/plugin"),
        primary_source_preflight=Path("/private/171.json"),
        replay_source_preflight=Path("/private/172.json"),
        artifact_paths=_paths(),
        checkpoint_dir=Path("/private/checkpoints"),
        environ={
            "QBOSON_USER_ID": "present",
            "QBOSON_SDK_CODE": "present",
            "QBOSON_PROJECT_NO": "present",
        },
        source_root=Path("/private/source"),
    )

    assert report["checks"]["source_preflights"] == {
        "status": "fail",
        "reason": "source_preflight_invalid",
    }
    assert report["ready_to_start_system_probe"] is False


def test_provider_smoke_readiness_does_not_require_protein_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sdk_approval = {"approved": True}
    monkeypatch.setattr(
        readiness_module,
        "verify_environment_lock",
        lambda path: ({"lock": True}, "c" * 64),
    )
    monkeypatch.setattr(
        readiness_module,
        "load_sdk_approval",
        lambda path: (sdk_approval, "1" * 64),
    )
    monkeypatch.setattr(
        readiness_module,
        "verify_approved_kaiwu_distribution",
        lambda environment, approval: None,
    )
    monkeypatch.setattr(readiness_module, "validate_private_directory", lambda *a, **k: None)

    report = audit_readiness(
        config_path=None,
        environment_lock_path=Path("/private/environment.json"),
        sdk_approval_path=Path("/private/sdk-approval.json"),
        plugin_root=None,
        primary_source_preflight=None,
        replay_source_preflight=None,
        artifact_paths=dict.fromkeys(_paths()),
        checkpoint_dir=Path("/private/checkpoints"),
        environ={
            "QBOSON_USER_ID": "present",
            "QBOSON_SDK_CODE": "present",
            "QBOSON_PROJECT_NO": "present",
        },
        required_stage="provider-smoke",
    )

    assert report["ready_to_start_provider_smoke"] is True
    assert report["ready_to_start_system_probe"] is False
    assert report["required_stage"] == "provider-smoke"
    assert report["required_stage_ready"] is True
    assert report["checks"]["config"]["status"] == "missing"


def test_readiness_rejects_unknown_required_stage() -> None:
    with pytest.raises(ValueError, match="required_stage"):
        audit_readiness(
            config_path=None,
            environment_lock_path=None,
            sdk_approval_path=None,
            plugin_root=None,
            primary_source_preflight=None,
            replay_source_preflight=None,
            artifact_paths=dict.fromkeys(_paths()),
            environ={},
            required_stage="unknown",
        )


@pytest.mark.parametrize(("ready", "expected_exit"), ((True, None), (False, 1)))
def test_readiness_cli_exit_tracks_selected_stage(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    ready: bool,
    expected_exit: int | None,
) -> None:
    captured: dict[str, object] = {}

    def _audit(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {
            "schema": readiness_module.SCHEMA,
            "required_stage": kwargs["required_stage"],
            "required_stage_ready": ready,
        }

    monkeypatch.setattr(readiness_module, "audit_readiness", _audit)
    monkeypatch.setattr(
        sys,
        "argv",
        ["audit_readiness", "--require-stage", "provider-smoke"],
    )

    if expected_exit is None:
        readiness_module.main()
    else:
        with pytest.raises(SystemExit) as raised:
            readiness_module.main()
        assert raised.value.code == expected_exit

    assert captured["required_stage"] == "provider-smoke"
    assert json.loads(capsys.readouterr().out)["required_stage_ready"] is ready


def test_system_readiness_rejects_different_sdk_approval_records(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    standalone_approval = {"approval_reference": "standalone"}
    config = {
        "software": {
            "source_revision": "a" * 40,
            "kaiwu_pytorch_plugin_revision": "b" * 40,
            "environment_lock_sha256": "c" * 64,
        },
        "kaiwu_sdk": {"approval_reference": "config"},
    }
    monkeypatch.setattr(
        readiness_module,
        "_load_frozen_config",
        lambda path: (config, "d" * 64),
    )
    monkeypatch.setattr(
        readiness_module,
        "build_quota_plan",
        lambda config, digest: {"totals": {"budget_complete": True}},
    )
    monkeypatch.setattr(
        readiness_module,
        "verify_environment_lock",
        lambda path: ({"lock": True}, "c" * 64),
    )
    monkeypatch.setattr(
        readiness_module,
        "load_sdk_approval",
        lambda path: (standalone_approval, "1" * 64),
    )
    monkeypatch.setattr(
        readiness_module,
        "verify_approved_kaiwu_distribution",
        lambda environment, approval: None,
    )
    monkeypatch.setattr(
        readiness_module,
        "load_source_preflight",
        lambda *args, **kwargs: ({"manifest_sha256": "e" * 64}, "f" * 64),
    )
    monkeypatch.setattr(
        readiness_module,
        "_inspect_artifacts",
        lambda config_path, artifact_paths: ("d" * 64, {}, {}),
    )
    monkeypatch.setattr(readiness_module, "validate_private_directory", lambda *a, **k: None)
    monkeypatch.setattr(readiness_module, "_inspect_a800_cuda_zero", lambda: "A800")

    report = audit_readiness(
        config_path=Path("/private/config.json"),
        environment_lock_path=Path("/private/environment.json"),
        sdk_approval_path=Path("/private/sdk-approval.json"),
        plugin_root=Path("/private/plugin"),
        primary_source_preflight=Path("/private/171.json"),
        replay_source_preflight=Path("/private/172.json"),
        artifact_paths=_paths(),
        checkpoint_dir=Path("/private/checkpoints"),
        environ={
            "QBOSON_USER_ID": "present",
            "QBOSON_SDK_CODE": "present",
            "QBOSON_PROJECT_NO": "present",
        },
        source_root=Path("/private/source"),
    )

    assert report["ready_to_start_provider_smoke"] is True
    assert report["checks"]["approval_alignment"] == {
        "status": "fail",
        "reason": "sdk_approvals_differ",
    }
    assert report["ready_to_start_system_probe"] is False


def test_system_readiness_requires_a800_cuda_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sdk_approval = {"approved": True}
    config = {
        "software": {
            "source_revision": "a" * 40,
            "kaiwu_pytorch_plugin_revision": "b" * 40,
            "environment_lock_sha256": "c" * 64,
        },
        "kaiwu_sdk": sdk_approval,
    }
    monkeypatch.setattr(
        readiness_module, "_load_frozen_config", lambda path: (config, "d" * 64)
    )
    monkeypatch.setattr(
        readiness_module,
        "build_quota_plan",
        lambda config, digest: {"totals": {"budget_complete": True}},
    )
    monkeypatch.setattr(
        readiness_module,
        "verify_environment_lock",
        lambda path: ({"lock": True}, "c" * 64),
    )
    monkeypatch.setattr(
        readiness_module,
        "load_sdk_approval",
        lambda path: (sdk_approval, "1" * 64),
    )
    monkeypatch.setattr(
        readiness_module,
        "verify_approved_kaiwu_distribution",
        lambda environment, approval: None,
    )
    monkeypatch.setattr(
        readiness_module,
        "load_source_preflight",
        lambda *args, **kwargs: ({"manifest_sha256": "e" * 64}, "f" * 64),
    )
    monkeypatch.setattr(
        readiness_module,
        "_inspect_artifacts",
        lambda config_path, artifact_paths: ("d" * 64, {}, {}),
    )
    monkeypatch.setattr(readiness_module, "validate_private_directory", lambda *a, **k: None)

    def _no_a800() -> str:
        raise RuntimeError("not an A800")

    monkeypatch.setattr(readiness_module, "_inspect_a800_cuda_zero", _no_a800)

    report = audit_readiness(
        config_path=Path("/private/config.json"),
        environment_lock_path=Path("/private/environment.json"),
        sdk_approval_path=Path("/private/sdk-approval.json"),
        plugin_root=Path("/private/plugin"),
        primary_source_preflight=Path("/private/171.json"),
        replay_source_preflight=Path("/private/172.json"),
        artifact_paths=_paths(),
        checkpoint_dir=Path("/private/checkpoints"),
        environ={
            "QBOSON_USER_ID": "present",
            "QBOSON_SDK_CODE": "present",
            "QBOSON_PROJECT_NO": "present",
        },
        source_root=Path("/private/source"),
        required_stage="system-probe",
    )

    assert report["ready_to_start_provider_smoke"] is True
    assert report["checks"]["a800_device"] == {
        "status": "fail",
        "reason": "a800_cuda_zero_unavailable",
    }
    assert report["ready_to_start_system_probe"] is False
    assert report["required_stage_ready"] is False
