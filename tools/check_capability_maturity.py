#!/usr/bin/env python
"""Validate the single source of truth for capability maturity claims."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "capability-maturity.toml"
EXPECTED_LEVELS = (
    "experimental",
    "development_evidence",
    "production_supported",
    "release_certified",
)
REQUIRED_USER_FIELDS = (
    "title",
    "summary",
    "category",
    "user_goals",
    "public_apis",
    "runtime_modes",
    "hardware",
    "gradient_support",
    "distribution_semantics",
    "quick_start",
    "documentation",
)
EXPECTED_CATEGORIES = (
    "build_and_compile",
    "simulation_and_training",
    "distributed_execution",
    "deployment_and_extension",
)

CLAIM_REQUIRED_FIELDS = (
    "id",
    "title",
    "artifact",
    "artifact_sha256",
    "code_version",
    "maturity",
    "scope",
    "environment_limitations",
)
CLAIM_VALUE_FORMATS = (
    "integer",
    "bytes",
    "seconds",
    "scientific",
    "string",
    "boolean",
)
CLAIM_AGGREGATES = ("single", "max", "min", "count", "unique")


def claim_values(payload: Any, selector: str) -> list[Any]:
    """Resolve a deliberately small JSON selector used by public claims."""
    values = [payload]
    for raw_part in selector.split("."):
        expand = raw_part.endswith("[]")
        part = raw_part[:-2] if expand else raw_part
        next_values: list[Any] = []
        for value in values:
            if not isinstance(value, dict) or part not in value:
                raise KeyError(selector)
            selected = value[part]
            if expand:
                if not isinstance(selected, list):
                    raise TypeError(selector)
                next_values.extend(selected)
            else:
                next_values.append(selected)
        values = next_values
    return values


def aggregate_claim_value(values: list[Any], aggregate: str) -> Any:
    if aggregate == "count":
        return len(values)
    if not values:
        raise ValueError("selector returned no values")
    if aggregate == "single":
        if len(values) != 1:
            raise ValueError(f"single requires one value, received {len(values)}")
        return values[0]
    if aggregate == "max":
        return max(values)
    if aggregate == "min":
        return min(values)
    if aggregate == "unique":
        unique = {json.dumps(value, sort_keys=True) for value in values}
        if len(unique) != 1:
            raise ValueError(
                f"unique requires one distinct value, received {len(unique)}"
            )
        return values[0]
    raise ValueError(f"unsupported aggregate {aggregate!r}")


def validate_claim_measurements(
    capability_name: str,
    claim_id: str,
    measurements: Any,
    payload: Any,
    field: str,
) -> list[str]:
    errors: list[str] = []
    if not isinstance(measurements, list) or not measurements:
        return [f"{capability_name}/{claim_id}: missing {field}"]
    for index, measurement in enumerate(measurements):
        prefix = f"{capability_name}/{claim_id}: {field}[{index}]"
        if not isinstance(measurement, dict):
            errors.append(f"{prefix} must be a table")
            continue
        for required in ("label", "selector", "format"):
            value = measurement.get(required)
            if not isinstance(value, str) or not value.strip():
                errors.append(f"{prefix} missing {required}")
        value_format = measurement.get("format")
        if value_format not in CLAIM_VALUE_FORMATS:
            errors.append(f"{prefix} has unsupported format {value_format!r}")
        aggregate = measurement.get("aggregate", "single")
        if aggregate not in CLAIM_AGGREGATES:
            errors.append(f"{prefix} has unsupported aggregate {aggregate!r}")
            continue
        selector = measurement.get("selector")
        if not isinstance(selector, str) or not selector:
            continue
        try:
            aggregate_claim_value(claim_values(payload, selector), aggregate)
        except (KeyError, TypeError, ValueError) as error:
            errors.append(f"{prefix} cannot resolve {selector!r}: {error}")
    return errors


def maturity_errors(data: dict[str, Any], root: Path = ROOT) -> tuple[str, ...]:
    errors: list[str] = []
    if data.get("schema") != "flagquantum_capability_maturity_v2":
        errors.append("unsupported capability maturity schema")
    levels = data.get("levels", {})
    if tuple(levels) != EXPECTED_LEVELS:
        errors.append(f"levels must be ordered as {EXPECTED_LEVELS!r}")
    ranks = [levels.get(name, {}).get("rank") for name in EXPECTED_LEVELS]
    if ranks != list(range(len(EXPECTED_LEVELS))):
        errors.append("maturity ranks must be contiguous from zero")

    capabilities = data.get("capabilities", {})
    if not capabilities:
        errors.append("at least one capability must be classified")
    claim_ids: set[str] = set()
    for name, capability in capabilities.items():
        for field in REQUIRED_USER_FIELDS:
            value = capability.get(field)
            if isinstance(value, str):
                valid = bool(value.strip())
            elif isinstance(value, list):
                valid = bool(value) and all(
                    isinstance(item, str) and item.strip() for item in value
                )
            else:
                valid = False
            if not valid:
                errors.append(f"{name}: missing or invalid user field {field}")
        if capability.get("category") not in EXPECTED_CATEGORIES:
            errors.append(
                f"{name}: unknown capability category {capability.get('category')!r}"
            )
        level = capability.get("level")
        if level not in levels:
            errors.append(f"{name}: unknown maturity level {level!r}")
            continue
        for field in levels[level].get("required_evidence", ()):
            value = capability.get(field)
            if not isinstance(value, str) or not value.strip():
                errors.append(f"{name}: {level} requires {field}")
        if level != "release_certified" and capability.get("release_gate"):
            errors.append(
                f"{name}: non-certified capability cannot declare release_gate"
            )
        for field in (
            "quick_start",
            "documentation",
            "focused_tests",
            "integration_tests",
            "hardware_evidence",
            "development_artifact",
            "operational_runbook",
            "release_gate",
            "release_artifact",
        ):
            value = capability.get(field)
            if isinstance(value, str) and not value.startswith("not_applicable"):
                if not (root / value).exists():
                    errors.append(f"{name}: {field} path does not exist: {value}")
        for claim in capability.get("performance_claims", ()):
            claim_id = (
                claim.get("id", "<missing>") if isinstance(claim, dict) else "<missing>"
            )
            if not isinstance(claim, dict):
                errors.append(f"{name}: performance claim must be a table")
                continue
            for field in CLAIM_REQUIRED_FIELDS:
                value = claim.get(field)
                if not isinstance(value, str) or not value.strip():
                    errors.append(f"{name}/{claim_id}: missing {field}")
            if claim_id in claim_ids:
                errors.append(f"{name}/{claim_id}: duplicate performance claim id")
            elif isinstance(claim_id, str):
                claim_ids.add(claim_id)
            if claim.get("maturity") != level:
                errors.append(
                    f"{name}/{claim_id}: claim maturity must equal capability "
                    f"level {level!r}"
                )
            artifact = claim.get("artifact")
            if not isinstance(artifact, str) or not artifact:
                continue
            artifact_path = root / artifact
            try:
                artifact_path.resolve().relative_to(
                    (root / "benchmarks/results").resolve()
                )
            except ValueError:
                errors.append(
                    f"{name}/{claim_id}: artifact must be under benchmarks/results"
                )
                continue
            if not artifact_path.is_file():
                errors.append(f"{name}/{claim_id}: artifact does not exist: {artifact}")
                continue
            digest = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
            if claim.get("artifact_sha256") != digest:
                errors.append(f"{name}/{claim_id}: artifact_sha256 mismatch")
            try:
                payload = json.loads(artifact_path.read_text(encoding="utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                errors.append(f"{name}/{claim_id}: artifact is not valid JSON: {error}")
                continue
            if payload.get("commit") != claim.get("code_version"):
                errors.append(
                    f"{name}/{claim_id}: code_version does not match artifact commit"
                )
            checks = claim.get("evidence_checks")
            if not isinstance(checks, dict) or not checks:
                errors.append(f"{name}/{claim_id}: missing evidence_checks")
            else:
                for selector, expected in checks.items():
                    try:
                        actual = aggregate_claim_value(
                            claim_values(payload, selector), "single"
                        )
                    except (KeyError, TypeError, ValueError) as error:
                        errors.append(
                            f"{name}/{claim_id}: evidence check cannot resolve "
                            f"{selector!r}: {error}"
                        )
                        continue
                    if actual != expected:
                        errors.append(
                            f"{name}/{claim_id}: evidence check failed for {selector!r}"
                        )
            errors.extend(
                validate_claim_measurements(
                    name,
                    str(claim_id),
                    claim.get("metrics"),
                    payload,
                    "metrics",
                )
            )
            errors.extend(
                validate_claim_measurements(
                    name,
                    str(claim_id),
                    claim.get("environment"),
                    payload,
                    "environment",
                )
            )
    return tuple(errors)


def main() -> int:
    data = tomllib.loads(MATRIX.read_text(encoding="utf-8"))
    errors = maturity_errors(data)
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print("capability maturity matrix passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
