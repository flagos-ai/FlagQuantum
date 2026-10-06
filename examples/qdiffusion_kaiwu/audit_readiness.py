"""Audit local QBoson/QDiffusion prerequisites without using credentials."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from examples.qdiffusion_kaiwu.plan_quota import build_quota_plan
from examples.qdiffusion_kaiwu.preflight_protein_artifacts import _inspect_artifacts
from examples.qdiffusion_kaiwu.private_io import validate_private_directory
from examples.qdiffusion_kaiwu.provider_resources import (
    assess_provider_resources,
    load_provider_resources,
)
from examples.qdiffusion_kaiwu.qdiffusion_system_live import _load_frozen_config
from examples.qdiffusion_kaiwu.sdk_approval import (
    load_sdk_approval,
    verify_approved_kaiwu_distribution,
)
from examples.qdiffusion_kaiwu.source_preflight import (
    load_source_preflight,
    validate_common_transfer_manifest,
)
from examples.qdiffusion_kaiwu.verify_environment_lock import (
    verify_environment_lock,
)

SCHEMA = "flagquantum.qboson_qdiffusion_readiness"
HOSTS = ("jp-a800-171", "jp-a800-172")
REQUIRED_STAGE_FIELDS = {
    "provider-smoke": "ready_to_start_provider_smoke",
    "system-probe": "ready_to_start_system_probe",
    "protein-experiment": "ready_to_start_protein_experiment",
}


def _check(status: str, reason: str) -> dict[str, str]:
    return {"status": status, "reason": reason}


def _valid_private_value(value: object) -> bool:
    return (
        isinstance(value, str) and bool(value.strip()) and value.strip().isprintable()
    )


def _inspect_a800_cuda_zero() -> str:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    device = torch.device("cuda:0")
    observed_gpu = torch.cuda.get_device_name(device)
    if "A800" not in observed_gpu:
        raise RuntimeError("cuda:0 is not an NVIDIA A800")
    return observed_gpu


def audit_readiness(
    *,
    config_path: Path | None,
    environment_lock_path: Path | None,
    sdk_approval_path: Path | None,
    plugin_root: Path | None,
    primary_source_preflight: Path | None,
    replay_source_preflight: Path | None,
    artifact_paths: dict[str, Path | None],
    provider_resources_path: Path | None = None,
    checkpoint_dir: Path | None = None,
    environ: Mapping[str, str] | None = None,
    source_root: Path | None = None,
    required_stage: str = "protein-experiment",
) -> dict[str, Any]:
    """Return a value-free, offline readiness report for the live runbook."""

    if required_stage not in REQUIRED_STAGE_FIELDS:
        raise ValueError("required_stage is not a supported readiness stage")
    environment = os.environ if environ is None else environ
    checks: dict[str, dict[str, str]] = {}
    user_id = environment.get("QBOSON_USER_ID")
    sdk_code = environment.get("QBOSON_SDK_CODE")
    if user_id is None and sdk_code is None:
        checks["credentials"] = _check("missing", "credential_pair_absent")
    elif user_id is None or sdk_code is None:
        checks["credentials"] = _check("fail", "partial_credential_pair")
    elif _valid_private_value(user_id) and _valid_private_value(sdk_code):
        checks["credentials"] = _check("pass", "credential_pair_present")
    else:
        checks["credentials"] = _check("fail", "credential_pair_invalid")
    project_no = environment.get("QBOSON_PROJECT_NO")
    if project_no is None:
        checks["project"] = _check("missing", "project_absent")
    elif _valid_private_value(project_no):
        checks["project"] = _check("pass", "project_present")
    else:
        checks["project"] = _check("fail", "project_invalid")

    if provider_resources_path is None:
        checks["provider_resources"] = _check(
            "missing", "provider_resource_snapshot_absent"
        )
    else:
        try:
            provider_resources, _ = load_provider_resources(provider_resources_path)
            resources_ready, resource_reason = assess_provider_resources(
                provider_resources
            )
        except (OSError, ValueError):
            checks["provider_resources"] = _check(
                "fail", "provider_resource_snapshot_invalid"
            )
        else:
            checks["provider_resources"] = _check(
                "pass" if resources_ready else "fail", resource_reason
            )

    if checkpoint_dir is None:
        checks["checkpoint_directory"] = _check(
            "missing", "checkpoint_directory_absent"
        )
    else:
        try:
            validate_private_directory(
                checkpoint_dir, label="Kaiwu checkpoint directory"
            )
        except (OSError, ValueError):
            checks["checkpoint_directory"] = _check(
                "fail", "checkpoint_directory_invalid"
            )
        else:
            checks["checkpoint_directory"] = _check(
                "pass", "checkpoint_directory_valid"
            )

    try:
        _inspect_a800_cuda_zero()
    except RuntimeError:
        checks["a800_device"] = _check("fail", "a800_cuda_zero_unavailable")
    else:
        checks["a800_device"] = _check("pass", "a800_cuda_zero_available")

    config: dict[str, Any] | None = None
    config_sha256: str | None = None
    if config_path is None:
        checks["config"] = _check("missing", "config_path_absent")
    else:
        try:
            config, config_sha256 = _load_frozen_config(config_path)
        except (OSError, ValueError):
            checks["config"] = _check("fail", "config_invalid")
        else:
            checks["config"] = _check("pass", "config_valid")

    quota_plan: dict[str, Any] | None = None
    if config is None or config_sha256 is None:
        checks["quota"] = _check("blocked", "config_not_validated")
    else:
        try:
            quota_plan = build_quota_plan(config, config_sha256)
        except ValueError:
            checks["quota"] = _check("fail", "quota_plan_invalid")
        else:
            checks["quota"] = (
                _check("pass", "quota_ceiling_complete")
                if quota_plan["totals"]["budget_complete"] is True
                else _check("fail", "quota_ceiling_unresolved")
            )

    environment_record: dict[str, Any] | None = None
    environment_sha256: str | None = None
    if environment_lock_path is None:
        checks["environment"] = _check("missing", "environment_lock_absent")
    else:
        try:
            environment_record, environment_sha256 = verify_environment_lock(
                environment_lock_path
            )
        except (OSError, ValueError):
            checks["environment"] = _check("fail", "environment_lock_invalid")
        else:
            checks["environment"] = _check("pass", "environment_lock_valid")

    sdk_approval: dict[str, Any] | None = None
    if environment_record is None:
        checks["sdk_approval"] = _check("blocked", "environment_not_validated")
    elif sdk_approval_path is None:
        checks["sdk_approval"] = _check("missing", "sdk_approval_absent")
    else:
        try:
            sdk_approval, _ = load_sdk_approval(sdk_approval_path)
            verify_approved_kaiwu_distribution(environment_record, sdk_approval)
        except (OSError, ValueError):
            checks["sdk_approval"] = _check("fail", "sdk_approval_invalid")
        else:
            checks["sdk_approval"] = _check("pass", "sdk_approval_valid")

    if sdk_approval is not None and checks["project"]["status"] == "pass":
        assert project_no is not None
        if project_no.strip() != sdk_approval.get("project_no"):
            checks["project"] = _check(
                "fail", "project_differs_from_reviewed_assignment"
            )
        else:
            checks["project"] = _check(
                "pass", "project_matches_reviewed_assignment"
            )

    if config is None:
        checks["frozen_environment"] = _check("blocked", "config_not_validated")
    elif environment_record is None or environment_sha256 is None:
        checks["frozen_environment"] = _check("blocked", "environment_not_validated")
    else:
        try:
            if environment_sha256 != config["software"]["environment_lock_sha256"]:
                raise ValueError("environment lock differs from frozen configuration")
            verify_approved_kaiwu_distribution(environment_record, config["kaiwu_sdk"])
        except (KeyError, ValueError):
            checks["frozen_environment"] = _check(
                "fail", "frozen_environment_or_sdk_approval_invalid"
            )
        else:
            checks["frozen_environment"] = _check(
                "pass", "frozen_environment_and_sdk_approval_valid"
            )

    if config is None:
        checks["approval_alignment"] = _check("blocked", "config_not_validated")
    elif sdk_approval is None:
        checks["approval_alignment"] = _check("blocked", "sdk_approval_not_validated")
    elif config.get("kaiwu_sdk") != sdk_approval:
        checks["approval_alignment"] = _check("fail", "sdk_approvals_differ")
    else:
        checks["approval_alignment"] = _check("pass", "sdk_approvals_match")

    preflight_paths = {
        "jp-a800-171": primary_source_preflight,
        "jp-a800-172": replay_source_preflight,
    }
    if config is None:
        checks["source_preflights"] = _check("blocked", "config_not_validated")
    elif plugin_root is None or any(path is None for path in preflight_paths.values()):
        checks["source_preflights"] = _check("missing", "source_inputs_absent")
    else:
        try:
            software = config["software"]
            execution_root = (
                Path(__file__).resolve().parents[2]
                if source_root is None
                else source_root
            )
            preflight_records: list[dict[str, Any]] = []
            for host in HOSTS:
                preflight_path = preflight_paths[host]
                assert preflight_path is not None
                record, _ = load_source_preflight(
                    preflight_path,
                    execution_host=host,
                    source_revision=software["source_revision"],
                    plugin_revision=software["kaiwu_pytorch_plugin_revision"],
                    source_root=execution_root,
                    plugin_root=plugin_root,
                )
                preflight_records.append(record)
            validate_common_transfer_manifest(
                (preflight_records[0], preflight_records[1])
            )
        except (KeyError, OSError, ValueError):
            checks["source_preflights"] = _check("fail", "source_preflight_invalid")
        else:
            checks["source_preflights"] = _check("pass", "both_source_preflights_valid")

    if config_path is None or config is None:
        checks["protein_artifacts"] = _check("blocked", "config_not_validated")
    elif set(artifact_paths) != {
        "dataset",
        "base_checkpoint",
        "tokenizer",
        "evaluation_model",
    } or any(path is None for path in artifact_paths.values()):
        checks["protein_artifacts"] = _check("missing", "artifact_paths_absent")
    else:
        try:
            _inspect_artifacts(
                config_path,
                {
                    name: path
                    for name, path in artifact_paths.items()
                    if path is not None
                },
            )
        except (OSError, ValueError):
            checks["protein_artifacts"] = _check("fail", "protein_artifacts_invalid")
        else:
            checks["protein_artifacts"] = _check("pass", "protein_artifacts_valid")

    provider_smoke_ready = all(
        checks[name]["status"] == "pass"
        for name in (
            "environment",
            "sdk_approval",
            "credentials",
            "project",
            "provider_resources",
            "checkpoint_directory",
        )
    )
    system_probe_ready = provider_smoke_ready and all(
        checks[name]["status"] == "pass"
        for name in (
            "config",
            "quota",
            "frozen_environment",
            "approval_alignment",
            "source_preflights",
            "a800_device",
        )
    )
    protein_experiment_ready = system_probe_ready and (
        checks["protein_artifacts"]["status"] == "pass"
    )
    readiness = {
        "ready_to_start_provider_smoke": provider_smoke_ready,
        "ready_to_start_system_probe": system_probe_ready,
        "ready_to_start_protein_experiment": protein_experiment_ready,
    }
    return {
        "schema": SCHEMA,
        "version": "1.0",
        "offline_readiness_only": True,
        "credential_values_recorded": False,
        "checks": checks,
        **readiness,
        "required_stage": required_stage,
        "required_stage_ready": readiness[REQUIRED_STAGE_FIELDS[required_stage]],
        "claim_boundary": (
            "Readiness is not provider, execution, hardware, or acceptance evidence."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--environment-lock", type=Path)
    parser.add_argument("--sdk-approval", type=Path)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--provider-resources", type=Path)
    parser.add_argument("--plugin-root", type=Path)
    parser.add_argument("--primary-source-preflight", type=Path)
    parser.add_argument("--replay-source-preflight", type=Path)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--base-checkpoint", type=Path)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--evaluation-model", type=Path)
    parser.add_argument(
        "--require-stage",
        choices=tuple(REQUIRED_STAGE_FIELDS),
        default="protein-experiment",
    )
    args = parser.parse_args()
    report = audit_readiness(
        config_path=args.config,
        environment_lock_path=args.environment_lock,
        sdk_approval_path=args.sdk_approval,
        plugin_root=args.plugin_root,
        primary_source_preflight=args.primary_source_preflight,
        replay_source_preflight=args.replay_source_preflight,
        artifact_paths={
            "dataset": args.dataset,
            "base_checkpoint": args.base_checkpoint,
            "tokenizer": args.tokenizer,
            "evaluation_model": args.evaluation_model,
        },
        provider_resources_path=args.provider_resources,
        checkpoint_dir=args.checkpoint_dir,
        required_stage=args.require_stage,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["required_stage_ready"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
