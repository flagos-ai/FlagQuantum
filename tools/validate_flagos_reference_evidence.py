#!/usr/bin/env python3
"""Fail closed when FlagOS CUDA-reference evidence exceeds its claim boundary."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOCK = ROOT / "ci" / "flagos_cuda_reference.lock.json"
DEFAULT_PROFILE = (
    ROOT / "flagquantum" / "runtime" / "profiles" / "statevector_local_p0.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _profile_identity(path: Path) -> tuple[str, int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    requirements = []
    for raw in payload["requirements"]:
        requirements.append(
            {
                "operator": raw["operator"],
                "dtypes": list(raw["dtypes"]),
                "forward": raw.get("forward", True),
                "backward": raw.get("backward", False),
                "deterministic": raw.get("deterministic", False),
            }
        )
    canonical = {
        "schema": payload.get("schema", "flagquantum_operator_profile_v1"),
        "profile_version": payload.get("profile_version", "1.0"),
        "name": payload["name"],
        "representation": payload["representation"],
        "distribution": payload["distribution"],
        "requirements": requirements,
    }
    encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest(), len(requirements)


def evidence_errors(
    payload: Mapping[str, Any],
    *,
    lock_path: Path = DEFAULT_LOCK,
    profile_path: Path = DEFAULT_PROFILE,
) -> tuple[str, ...]:
    """Return stable blocker strings for malformed or over-claimed evidence."""

    errors: list[str] = []
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("schema") != "flagquantum_flagos_environment_lock_v2":
        errors.append("environment_lock_schema_unsupported")
    profile_hash, requirement_count = _profile_identity(profile_path)
    expected_pairs = {
        (str(dtype), int(depth))
        for dtype in lock["runtime_contract"]["dtypes"]
        for depth in lock["runtime_contract"]["depths"]
    }

    expected_scalars = {
        "schema": "flagquantum_flagos_cuda_reference_evidence_v2",
        "status": "passed",
        "validation_scope": "flagos_cuda_reference",
        "artifact_class": "development_reference",
        "distribution_semantics": "single_device_fast_path",
        "hardware_certification": False,
        "scalability_claim_allowed": False,
        "device": lock["runtime_contract"]["logical_device"],
        "device_count": lock["runtime_contract"]["visible_device_count"],
        "operator_profile": lock["runtime_contract"]["operator_profile"],
        "operator_profile_hash": profile_hash,
        "operator_evidence_count": requirement_count,
    }
    for field, expected in expected_scalars.items():
        if payload.get(field) != expected:
            errors.append(f"{field}_mismatch")

    lock_evidence = payload.get("environment_lock")
    if not isinstance(lock_evidence, Mapping):
        errors.append("environment_lock_missing")
    else:
        if lock_evidence.get("sha256") != _sha256(lock_path):
            errors.append("environment_lock_hash_mismatch")
        if lock_evidence.get("schema") != lock.get("schema"):
            errors.append("environment_lock_schema_mismatch")

    environment = payload.get("environment")
    expected_environment = {
        "container_image": lock["container_image"],
        "python": lock["python"],
        "torch_package": lock["torch"]["package"],
        "torch_distribution": lock["torch"]["distribution"],
        "torch_cuda_runtime": lock["torch"]["cuda_runtime"],
        "torch_fl_package": lock["torch_fl"]["package"],
        "torch_fl_commit": lock["torch_fl"]["commit"],
        "torch_fl_build": lock["torch_fl"]["build"],
    }
    if not isinstance(environment, Mapping):
        errors.append("environment_identity_missing")
    else:
        if environment.get("physical_accelerator") in {None, "", "unavailable"}:
            errors.append("environment_physical_accelerator_missing")
        for field, expected in expected_environment.items():
            if environment.get(field) != expected:
                errors.append(f"environment_{field}_mismatch")

    source = payload.get("source")
    if not isinstance(source, Mapping) or source.get("revision") in {
        None,
        "",
        "unavailable",
    }:
        errors.append("source_revision_missing")
    elif source.get("tree_dirty") is not False:
        errors.append("source_tree_dirty")

    blockers = set(payload.get("claim_blockers", ()))
    required_blockers = {
        "domestic_accelerator_hardware_not_exercised",
        "route_fallback_absence_not_certified",
        "production_performance_not_measured",
    }
    if not required_blockers.issubset(blockers):
        errors.append("claim_boundary_incomplete")

    matrix = payload.get("validation_matrix")
    if not isinstance(matrix, list):
        errors.append("validation_matrix_missing")
        return tuple(sorted(set(errors)))
    actual_pairs: set[tuple[str, int]] = set()
    for item in matrix:
        if not isinstance(item, Mapping):
            errors.append("validation_matrix_item_invalid")
            continue
        pair = (str(item.get("dtype")), int(item.get("depth", -1)))
        actual_pairs.add(pair)
        metrics = item.get("state_metrics")
        numerical = item.get("numerical_validation")
        if not isinstance(metrics, Mapping):
            errors.append(f"state_metrics_missing:{pair[0]}:{pair[1]}")
            continue
        if not isinstance(numerical, Mapping) or numerical.get("passed") is not True:
            errors.append(f"numerical_validation_failed:{pair[0]}:{pair[1]}")
        tolerance = 2e-5 if pair[0] == "complex64" else 1e-11
        for field in ("max_abs_error", "norm_drift", "state_infidelity"):
            value = metrics.get(field)
            if not isinstance(value, (int, float)) or not 0 <= value <= tolerance:
                errors.append(f"{field}_out_of_bounds:{pair[0]}:{pair[1]}")
    if actual_pairs != expected_pairs or len(matrix) != len(expected_pairs):
        errors.append("validation_matrix_coverage_mismatch")
    return tuple(sorted(set(errors)))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    args = parser.parse_args(argv)
    payload = json.loads(args.evidence.read_text(encoding="utf-8"))
    errors = evidence_errors(payload, lock_path=args.lock, profile_path=args.profile)
    if errors:
        print("\n".join(errors))
        return 1
    print("FlagOS CUDA-reference evidence passed its development-only gate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
