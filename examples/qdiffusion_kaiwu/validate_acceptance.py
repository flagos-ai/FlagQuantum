"""Fail-closed validator for QBoson QDiffusion acceptance evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any

HOSTS = {"jp-a800-171", "jp-a800-172"}
FULL_REVISION = re.compile(r"[0-9a-f]{40}")
CONFIG_SCHEMA = "flagquantum.qboson_qdiffusion_config"
RECORD_SCHEMA = "flagquantum.qboson_qdiffusion_acceptance"
MANIFEST_SCHEMA = "flagquantum.qboson_qdiffusion_manifest"


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mapping(value: Any, label: str, errors: list[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        errors.append(f"{label}: expected an object")
        return {}
    return value


def _finite_number(value: Any, label: str, errors: list[str]) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        errors.append(f"{label}: expected a finite number")
        return None
    number = float(value)
    if not math.isfinite(number):
        errors.append(f"{label}: expected a finite number")
        return None
    return number


def _validate_config(config: dict[str, Any], errors: list[str]) -> None:
    if config.get("schema") != CONFIG_SCHEMA or config.get("version") != "1.0":
        errors.append("config: unsupported schema or version")
    primary_host = config.get("primary_host")
    replay_host = config.get("replay_host")
    if primary_host not in HOSTS:
        errors.append("config.primary_host: expected one declared A800 host")
    if replay_host not in HOSTS or replay_host == primary_host:
        errors.append("config.replay_host: expected the other declared A800 host")
    preregistered_at = config.get("preregistered_at")
    try:
        timestamp = datetime.fromisoformat(str(preregistered_at).replace("Z", "+00:00"))
        if timestamp.tzinfo is None:
            raise ValueError
    except ValueError:
        errors.append("config.preregistered_at: expected a timezone-aware timestamp")
    seeds = config.get("seeds")
    if (
        not isinstance(seeds, list)
        or len(seeds) < 3
        or any(type(seed) is not int for seed in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        errors.append("config.seeds: expected at least three unique integer seeds")
    for name in ("dataset", "checkpoint", "tokenizer", "generation"):
        section = _mapping(config.get(name), f"config.{name}", errors)
        if not section or any(
            value is None or value == "" or value == "<required>"
            for value in section.values()
        ):
            errors.append(f"config.{name}: all frozen identity fields are required")
    for name in ("dataset", "checkpoint"):
        digest = _mapping(config.get(name), f"config.{name}", errors).get("sha256")
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            errors.append(f"config.{name}.sha256: expected a SHA-256 digest")
    generation = _mapping(config.get("generation"), "config.generation", errors)
    for field in ("sequence_count", "max_steps", "num_candidates"):
        if type(generation.get(field)) is not int or generation.get(field, 0) <= 0:
            errors.append(f"config.generation.{field}: expected a positive integer")
    software = _mapping(config.get("software"), "config.software", errors)
    for field in (
        "source_revision",
        "kaiwu_pytorch_plugin_revision",
        "python_version",
        "torch_version",
        "kaiwu_sdk_version",
    ):
        if software.get(field) in {None, "", "<required>"}:
            errors.append(f"config.software.{field}: frozen value is required")
    for field in ("source_revision", "kaiwu_pytorch_plugin_revision"):
        revision = software.get(field)
        if not isinstance(revision, str) or FULL_REVISION.fullmatch(revision) is None:
            errors.append(f"config.software.{field}: expected a full Git revision")
    precision = _mapping(
        config.get("precision_policy"), "config.precision_policy", errors
    )
    for field in ("name", "target_min", "target_max"):
        if field not in precision:
            errors.append(f"config.precision_policy: missing {field}")
    if config.get("primary_metric") != {
        "name": "mean_cosine_distance",
        "direction": "lower",
    }:
        errors.append(
            "config.primary_metric: expected preregistered lower mean_cosine_distance"
        )
    if (
        type(config.get("remote_call_budget")) is not int
        or config.get("remote_call_budget", 0) <= 0
    ):
        errors.append("config.remote_call_budget: expected a positive integer")
    thresholds = _mapping(config.get("thresholds"), "config.thresholds", errors)
    if thresholds.get("uniqueness_baseline_fraction_min") != 0.95:
        errors.append("config.thresholds: uniqueness floor must remain 0.95")
    if thresholds.get("repeat_ratio_absolute_increase_max") != 0.05:
        errors.append("config.thresholds: repeat-ratio allowance must remain 0.05")
    if thresholds.get("invalid_sequence_count_max") != 0:
        errors.append("config.thresholds: invalid sequence count must remain zero")


def _validate_system_record(
    record: dict[str, Any],
    *,
    config: dict[str, Any],
    config_sha256: str,
    label: str,
    errors: list[str],
) -> None:
    if record.get("schema") != RECORD_SCHEMA or record.get("version") != "1.0":
        errors.append(f"{label}: unsupported schema or version")
    for field in ("source_revision", "kaiwu_pytorch_plugin_revision"):
        value = record.get(field)
        if not isinstance(value, str) or FULL_REVISION.fullmatch(value) is None:
            errors.append(f"{label}.{field}: expected a full Git revision")
    software = _mapping(config.get("software"), "config.software", errors)
    for field in (
        "source_revision",
        "kaiwu_pytorch_plugin_revision",
        "python_version",
        "torch_version",
        "kaiwu_sdk_version",
    ):
        if record.get(field) != software.get(field):
            errors.append(f"{label}.{field}: differs from the frozen software lane")
    if record.get("experiment_config_sha256") != config_sha256:
        errors.append(f"{label}: frozen experiment config identity mismatch")
    if record.get("execution_host") not in HOSTS:
        errors.append(f"{label}.execution_host: unexpected host")
    if record.get("run_role") not in {"primary", "portability_replay"}:
        errors.append(f"{label}.run_role: unexpected role")
    if record.get("requested_cuda_device") != "cuda:0":
        errors.append(f"{label}: acceptance requires explicit cuda:0")
    if record.get("observed_tensor_device") != "cuda:0":
        errors.append(f"{label}: tensor work was not observed on cuda:0")
    if "A800" not in str(record.get("observed_gpu_model", "")):
        errors.append(f"{label}: observed GPU is not an NVIDIA A800")
    if record.get("transport") != "kaiwu_cim":
        errors.append(f"{label}: transport is not kaiwu_cim")
    if record.get("qboson_hardware_used") is not True:
        errors.append(f"{label}: QBoson hardware use is not proven")
    if record.get("real_provider_evidence") is not True:
        errors.append(f"{label}: real provider evidence is absent")
    if record.get("provider_reported_target") is not True:
        errors.append(f"{label}: target identity is not provider-reported")
    if (
        not isinstance(record.get("qboson_target"), str)
        or not record["qboson_target"].strip()
    ):
        errors.append(f"{label}: provider target is missing")
    task_ids = record.get("qboson_task_ids")
    if (
        not isinstance(task_ids, list)
        or not task_ids
        or any(
            not isinstance(task_id, str) or not task_id.strip() for task_id in task_ids
        )
    ):
        errors.append(f"{label}: provider task identities are missing")
    if record.get("sampling_mode") != "sampling":
        errors.append(f"{label}: expected sampling mode")
    requested = _finite_number(
        record.get("requested_samples"), f"{label}.requested_samples", errors
    )
    returned = _finite_number(
        record.get("returned_samples"), f"{label}.returned_samples", errors
    )
    if requested is not None and returned is not None and requested != returned:
        errors.append(f"{label}: requested and returned sample counts differ")
    call_count = record.get("remote_call_count")
    call_budget = record.get("remote_call_budget")
    if (
        type(call_count) is not int
        or type(call_budget) is not int
        or call_count <= 0
        or call_count > call_budget
    ):
        errors.append(f"{label}: remote-call accounting violates its budget")
    if record.get("fallback_occurred") is not False:
        errors.append(f"{label}: fallback must be explicitly false")
    if record.get("retrieval_resubmitted") is not False:
        errors.append(f"{label}: retrieval must not resubmit")
    if record.get("secrets_redacted") is not True:
        errors.append(f"{label}: secret redaction is not proven")
    precision = _mapping(
        record.get("precision_policy"), f"{label}.precision_policy", errors
    )
    expected_precision = _mapping(
        config.get("precision_policy"), "config.precision_policy", errors
    )
    for field in ("name", "target_min", "target_max"):
        if precision.get(field) != expected_precision.get(field):
            errors.append(f"{label}.precision_policy.{field}: differs from config")
    for field in ("name", "scale_factor", "target_min", "target_max", "max_abs_error"):
        if field not in precision:
            errors.append(f"{label}.precision_policy: missing {field}")
    training = _mapping(record.get("training"), f"{label}.training", errors)
    for field in ("energy_objective", "gradient_norm", "parameter_delta_max"):
        value = _finite_number(training.get(field), f"{label}.training.{field}", errors)
        if field != "energy_objective" and value is not None and value <= 0:
            errors.append(f"{label}.training.{field}: expected a positive value")
    generation = _mapping(record.get("generation"), f"{label}.generation", errors)
    if generation.get("token_constraints_passed") is not True:
        errors.append(f"{label}: token constraints did not pass")
    if generation.get("invalid_sequence_count") != 0:
        errors.append(f"{label}: invalid generated sequences were observed")
    acceptance = _mapping(record.get("acceptance"), f"{label}.acceptance", errors)
    if acceptance.get("system") != "pass":
        errors.append(f"{label}: system acceptance did not pass")


def _metric(
    metrics: dict[str, Any], name: str, label: str, errors: list[str]
) -> float | None:
    return _finite_number(metrics.get(name), f"{label}.{name}", errors)


def _validate_application(
    primary: dict[str, Any], config: dict[str, Any], errors: list[str]
) -> None:
    baseline = _mapping(
        primary.get("baseline_metrics"), "primary.baseline_metrics", errors
    )
    guided = _mapping(primary.get("guided_metrics"), "primary.guided_metrics", errors)
    metric_names = (
        "mean_cosine_distance",
        "median_cosine_distance",
        "mean_l2_distance",
        "median_l2_distance",
        "identity_to_reference_mean",
        "amino_acid_jsd",
        "kmer2_jsd",
        "kmer3_jsd",
        "uniqueness_ratio",
        "repeat_ratio_ge4",
        "length_match_ratio",
        "invalid_sequence_count",
    )
    baseline_values = {
        name: _metric(baseline, name, "primary.baseline_metrics", errors)
        for name in metric_names
    }
    guided_values = {
        name: _metric(guided, name, "primary.guided_metrics", errors)
        for name in metric_names
    }
    baseline_primary = baseline_values["mean_cosine_distance"]
    guided_primary = guided_values["mean_cosine_distance"]
    if (
        baseline_primary is not None
        and guided_primary is not None
        and guided_primary >= baseline_primary
    ):
        errors.append("primary: guided mean cosine distance did not improve")
    baseline_unique = baseline_values["uniqueness_ratio"]
    guided_unique = guided_values["uniqueness_ratio"]
    thresholds = _mapping(config.get("thresholds"), "config.thresholds", errors)
    uniqueness_floor = _finite_number(
        thresholds.get("uniqueness_baseline_fraction_min"),
        "config.thresholds.uniqueness_baseline_fraction_min",
        errors,
    )
    if (
        baseline_unique is not None
        and guided_unique is not None
        and uniqueness_floor is not None
        and guided_unique < uniqueness_floor * baseline_unique
    ):
        errors.append("primary: guided uniqueness is below its preregistered floor")
    baseline_repeat = baseline_values["repeat_ratio_ge4"]
    guided_repeat = guided_values["repeat_ratio_ge4"]
    repeat_allowance = _finite_number(
        thresholds.get("repeat_ratio_absolute_increase_max"),
        "config.thresholds.repeat_ratio_absolute_increase_max",
        errors,
    )
    if (
        baseline_repeat is not None
        and guided_repeat is not None
        and repeat_allowance is not None
        and guided_repeat > baseline_repeat + repeat_allowance
    ):
        errors.append("primary: guided repeat ratio exceeds its preregistered limit")
    if guided_values["invalid_sequence_count"] != 0:
        errors.append("primary: guided generation contains invalid sequences")
    if primary.get("attempted_seeds") != config.get("seeds"):
        errors.append("primary: attempted seeds differ from the frozen seed set")
    acceptance = _mapping(primary.get("acceptance"), "primary.acceptance", errors)
    if acceptance.get("application") != "pass":
        errors.append("primary: application acceptance did not pass")


def validate_acceptance(manifest_path: Path) -> list[str]:
    """Return all validation errors; an empty list means the evidence passes."""

    errors: list[str] = []
    root = manifest_path.resolve().parent
    manifest = _mapping(_read_json(manifest_path), "manifest", errors)
    if manifest.get("schema") != MANIFEST_SCHEMA or manifest.get("version") != "1.0":
        errors.append("manifest: unsupported schema or version")

    def resolve_member(value: Any, label: str) -> Path | None:
        if not isinstance(value, str) or not value:
            errors.append(f"{label}: expected a relative path")
            return None
        member = (root / value).resolve()
        if Path(value).is_absolute() or not member.is_relative_to(root):
            errors.append(f"{label}: path escapes the evidence directory")
            return None
        if not member.is_file():
            errors.append(f"{label}: file does not exist")
            return None
        return member

    config_entry = _mapping(manifest.get("config"), "manifest.config", errors)
    config_path = resolve_member(config_entry.get("path"), "manifest.config.path")
    if config_path is None:
        return errors
    config_hash = _sha256(config_path)
    if config_entry.get("sha256") != config_hash:
        errors.append("manifest.config: SHA-256 mismatch")
    config = _mapping(_read_json(config_path), "config", errors)
    _validate_config(config, errors)

    entries = manifest.get("records")
    if not isinstance(entries, list) or len(entries) != 2:
        errors.append("manifest.records: expected exactly two host records")
        return errors
    records: list[dict[str, Any]] = []
    for index, raw_entry in enumerate(entries):
        entry = _mapping(raw_entry, f"manifest.records[{index}]", errors)
        record_path = resolve_member(
            entry.get("path"), f"manifest.records[{index}].path"
        )
        if record_path is None:
            continue
        if entry.get("sha256") != _sha256(record_path):
            errors.append(f"manifest.records[{index}]: SHA-256 mismatch")
        record = _mapping(_read_json(record_path), f"record[{index}]", errors)
        _validate_system_record(
            record,
            config=config,
            config_sha256=config_hash,
            label=f"record[{index}]",
            errors=errors,
        )
        records.append(record)
    if len(records) != 2:
        return errors

    by_host = {str(record.get("execution_host")): record for record in records}
    if set(by_host) != HOSTS:
        errors.append("manifest: records must cover jp-a800-171 and jp-a800-172")
        return errors
    primary_host = config.get("primary_host")
    replay_host = config.get("replay_host")
    if primary_host not in by_host or replay_host not in by_host:
        errors.append("manifest: configured host roles do not match the records")
        return errors
    primary = by_host[primary_host]
    replay = by_host[replay_host]
    if primary.get("run_role") != "primary":
        errors.append("manifest: configured primary host lacks the primary role")
    if replay.get("run_role") != "portability_replay":
        errors.append("manifest: configured replay host lacks the portability role")
    if (
        _mapping(replay.get("acceptance"), "replay.acceptance", errors).get(
            "application"
        )
        != "not_run"
    ):
        errors.append("replay: application acceptance must be explicitly not_run")
    revisions = {record.get("source_revision") for record in records}
    plugin_revisions = {
        record.get("kaiwu_pytorch_plugin_revision") for record in records
    }
    if len(revisions) != 1 or len(plugin_revisions) != 1:
        errors.append("manifest: both hosts must use identical source revisions")
    task_sets = []
    for record in records:
        raw_task_ids = record.get("qboson_task_ids")
        task_sets.append(
            {task_id for task_id in raw_task_ids if isinstance(task_id, str)}
            if isinstance(raw_task_ids, list)
            else set()
        )
    if task_sets[0] & task_sets[1]:
        errors.append(
            "manifest: host runs must have independent QBoson task identities"
        )
    if any(
        record.get("remote_call_budget") != config.get("remote_call_budget")
        for record in records
    ):
        errors.append("manifest: host call budgets differ from the frozen config")
    _validate_application(primary, config, errors)
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    arguments = parser.parse_args()
    errors = validate_acceptance(arguments.manifest)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        raise SystemExit(1)
    print("QBoson QDiffusion acceptance evidence passed")


if __name__ == "__main__":
    main()
