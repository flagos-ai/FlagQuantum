"""Audit local QBoson/QDiffusion prerequisites without using credentials."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from examples.qdiffusion_kaiwu.plan_quota import build_quota_plan
from examples.qdiffusion_kaiwu.preflight_protein_artifacts import _inspect_artifacts
from examples.qdiffusion_kaiwu.qdiffusion_system_live import _load_frozen_config
from examples.qdiffusion_kaiwu.sdk_approval import (
    verify_approved_kaiwu_distribution,
)
from examples.qdiffusion_kaiwu.source_preflight import (
    load_source_preflight,
    validate_common_transfer_manifest,
)
from examples.qdiffusion_kaiwu.verify_environment_lock import (
    verify_frozen_environment_lock,
)

SCHEMA = "flagquantum.qboson_qdiffusion_readiness"
HOSTS = ("jp-a800-171", "jp-a800-172")


def _check(status: str, reason: str) -> dict[str, str]:
    return {"status": status, "reason": reason}


def audit_readiness(
    *,
    config_path: Path | None,
    environment_lock_path: Path | None,
    plugin_root: Path | None,
    primary_source_preflight: Path | None,
    replay_source_preflight: Path | None,
    artifact_paths: dict[str, Path | None],
    environ: Mapping[str, str] | None = None,
    source_root: Path | None = None,
) -> dict[str, Any]:
    """Return a value-free, offline readiness report for the live runbook."""

    environment = os.environ if environ is None else environ
    checks: dict[str, dict[str, str]] = {}
    user_present = bool(environment.get("QBOSON_USER_ID"))
    code_present = bool(environment.get("QBOSON_SDK_CODE"))
    if user_present and code_present:
        checks["credentials"] = _check("pass", "credential_pair_present")
    elif user_present or code_present:
        checks["credentials"] = _check("fail", "partial_credential_pair")
    else:
        checks["credentials"] = _check("missing", "credential_pair_absent")
    checks["project"] = (
        _check("pass", "project_present")
        if bool(environment.get("QBOSON_PROJECT_NO"))
        else _check("missing", "project_absent")
    )

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

    if config is None:
        checks["environment"] = _check("blocked", "config_not_validated")
    elif environment_lock_path is None:
        checks["environment"] = _check("missing", "environment_lock_absent")
    else:
        try:
            environment_record, _ = verify_frozen_environment_lock(
                environment_lock_path,
                expected_sha256=config["software"]["environment_lock_sha256"],
            )
            verify_approved_kaiwu_distribution(
                environment_record, config["kaiwu_sdk"]
            )
        except (KeyError, OSError, ValueError):
            checks["environment"] = _check(
                "fail", "environment_or_sdk_approval_invalid"
            )
        else:
            checks["environment"] = _check(
                "pass", "environment_and_sdk_approval_valid"
            )

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
            checks["source_preflights"] = _check(
                "fail", "source_preflight_invalid"
            )
        else:
            checks["source_preflights"] = _check(
                "pass", "both_source_preflights_valid"
            )

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
                {name: path for name, path in artifact_paths.items() if path is not None},
            )
        except (OSError, ValueError):
            checks["protein_artifacts"] = _check(
                "fail", "protein_artifacts_invalid"
            )
        else:
            checks["protein_artifacts"] = _check(
                "pass", "protein_artifacts_valid"
            )

    provider_smoke_ready = all(
        checks[name]["status"] == "pass"
        for name in ("config", "environment", "credentials", "project")
    )
    system_probe_ready = provider_smoke_ready and all(
        checks[name]["status"] == "pass"
        for name in ("quota", "source_preflights")
    )
    protein_experiment_ready = system_probe_ready and (
        checks["protein_artifacts"]["status"] == "pass"
    )
    return {
        "schema": SCHEMA,
        "version": "1.0",
        "offline_readiness_only": True,
        "credential_values_recorded": False,
        "checks": checks,
        "ready_to_start_provider_smoke": provider_smoke_ready,
        "ready_to_start_system_probe": system_probe_ready,
        "ready_to_start_protein_experiment": protein_experiment_ready,
        "claim_boundary": (
            "Readiness is not provider, execution, hardware, or acceptance evidence."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--environment-lock", type=Path)
    parser.add_argument("--plugin-root", type=Path)
    parser.add_argument("--primary-source-preflight", type=Path)
    parser.add_argument("--replay-source-preflight", type=Path)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--base-checkpoint", type=Path)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--evaluation-model", type=Path)
    args = parser.parse_args()
    report = audit_readiness(
        config_path=args.config,
        environment_lock_path=args.environment_lock,
        plugin_root=args.plugin_root,
        primary_source_preflight=args.primary_source_preflight,
        replay_source_preflight=args.replay_source_preflight,
        artifact_paths={
            "dataset": args.dataset,
            "base_checkpoint": args.base_checkpoint,
            "tokenizer": args.tokenizer,
            "evaluation_model": args.evaluation_model,
        },
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["ready_to_start_protein_experiment"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
