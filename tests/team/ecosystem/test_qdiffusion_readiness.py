from __future__ import annotations

import json
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
    assert report["ready_to_start_provider_smoke"] is False
    assert report["ready_to_start_system_probe"] is False
    assert report["ready_to_start_protein_experiment"] is False
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

    report = audit_readiness(
        config_path=Path("/private/config.json"),
        environment_lock_path=Path("/private/environment.json"),
        sdk_approval_path=Path("/private/sdk-approval.json"),
        plugin_root=Path("/private/plugin"),
        primary_source_preflight=Path("/private/171.json"),
        replay_source_preflight=Path("/private/172.json"),
        artifact_paths=_paths(),
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

    report = audit_readiness(
        config_path=Path("/private/config.json"),
        environment_lock_path=Path("/private/environment.json"),
        sdk_approval_path=Path("/private/sdk-approval.json"),
        plugin_root=Path("/private/plugin"),
        primary_source_preflight=Path("/private/171.json"),
        replay_source_preflight=Path("/private/172.json"),
        artifact_paths=_paths(),
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

    report = audit_readiness(
        config_path=None,
        environment_lock_path=Path("/private/environment.json"),
        sdk_approval_path=Path("/private/sdk-approval.json"),
        plugin_root=None,
        primary_source_preflight=None,
        replay_source_preflight=None,
        artifact_paths=dict.fromkeys(_paths()),
        environ={
            "QBOSON_USER_ID": "present",
            "QBOSON_SDK_CODE": "present",
            "QBOSON_PROJECT_NO": "present",
        },
    )

    assert report["ready_to_start_provider_smoke"] is True
    assert report["ready_to_start_system_probe"] is False
    assert report["checks"]["config"]["status"] == "missing"


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

    report = audit_readiness(
        config_path=Path("/private/config.json"),
        environment_lock_path=Path("/private/environment.json"),
        sdk_approval_path=Path("/private/sdk-approval.json"),
        plugin_root=Path("/private/plugin"),
        primary_source_preflight=Path("/private/171.json"),
        replay_source_preflight=Path("/private/172.json"),
        artifact_paths=_paths(),
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
