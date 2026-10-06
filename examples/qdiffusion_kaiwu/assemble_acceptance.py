"""Assemble immutable QDiffusion component records into final host evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
from pathlib import Path
from statistics import fmean
from tempfile import TemporaryDirectory
from typing import Any

from examples.qdiffusion_kaiwu.preflight_protein_artifacts import (
    PREFLIGHT_SCHEMA as ARTIFACT_PREFLIGHT_SCHEMA,
)
from examples.qdiffusion_kaiwu.private_io import read_private_bytes
from examples.qdiffusion_kaiwu.qboson_live_smoke import SCHEMA as PROVIDER_SMOKE_SCHEMA
from examples.qdiffusion_kaiwu.qdiffusion_portability_replay_live import (
    SCHEMA as PORTABILITY_SCHEMA,
)
from examples.qdiffusion_kaiwu.qdiffusion_protein_evaluate import (
    SCHEMA as EVALUATION_SCHEMA,
)
from examples.qdiffusion_kaiwu.qdiffusion_protein_training_live import (
    SCHEMA as TRAINING_SCHEMA,
)
from examples.qdiffusion_kaiwu.qdiffusion_system_live import (
    SCHEMA as SYSTEM_SCHEMA,
)
from examples.qdiffusion_kaiwu.qdiffusion_system_live import _load_frozen_config
from examples.qdiffusion_kaiwu.sdk_approval import SCHEMA as SDK_APPROVAL_SCHEMA
from examples.qdiffusion_kaiwu.source_preflight import (
    SCHEMA as SOURCE_PREFLIGHT_SCHEMA,
)
from examples.qdiffusion_kaiwu.source_preflight import (
    TRANSFER_MANIFEST_SCHEMA,
    validate_common_transfer_manifest,
    validate_source_preflight_record,
    validate_transfer_manifest_record,
)
from examples.qdiffusion_kaiwu.strict_json import loads_json_strict
from examples.qdiffusion_kaiwu.validate_acceptance import (
    MANIFEST_SCHEMA,
    RECORD_SCHEMA,
    _validate_application,
    _validate_evaluation_component,
    _validate_portability_component_evidence,
    _validate_remote_sampling_component_evidence,
    _validate_system_record,
    _validate_training_component,
    validate_acceptance,
)

METRIC_NAMES = (
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

_MAX_CONFIG_BYTES = 1024 * 1024
_MAX_ENVIRONMENT_LOCK_BYTES = 4 * 1024 * 1024
_MAX_COMPONENT_BYTES = 64 * 1024 * 1024


def _require_private_directory(path: Path, *, description: str) -> None:
    try:
        metadata = path.lstat()
    except OSError:
        raise ValueError(
            f"{description} must be an existing private, non-symlink directory: {path}"
        ) from None
    if (
        path.is_symlink()
        or not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_mode & 0o077
    ):
        raise ValueError(
            f"{description} must be an existing private, non-symlink directory: {path}"
        )


def _require_private_input(path: Path) -> None:
    if not path.is_absolute():
        raise ValueError(f"evidence input must use an absolute path: {path}")
    _require_private_directory(path.parent, description="evidence input parent")
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"evidence input must be a regular, non-symlink file: {path}")
    if path.stat().st_mode & 0o077:
        raise ValueError(
            f"evidence input must not be accessible by group or others: {path}"
        )


def _load_component(path: Path, schema: str) -> tuple[dict[str, Any], str]:
    encoded = read_private_bytes(
        path,
        label="component record",
        max_bytes=_MAX_COMPONENT_BYTES,
    )
    value = loads_json_strict(encoded)
    if not isinstance(value, dict):
        raise ValueError(f"component record must be a JSON object: {path}")
    if value.get("schema") != schema or value.get("version") != "1.0":
        raise ValueError(f"component record has an unsupported schema: {path}")
    return value, hashlib.sha256(encoded).hexdigest()


def _finite_metric(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite number")
    return result


def _aggregate_metrics(
    evaluation_records: list[dict[str, Any]], field: str
) -> dict[str, float]:
    output: dict[str, float] = {}
    for metric in METRIC_NAMES:
        values = [
            _finite_metric(record[field].get(metric), f"{field}.{metric}")
            for record in evaluation_records
            if isinstance(record.get(field), dict)
        ]
        if len(values) != len(evaluation_records):
            raise ValueError(f"an evaluation record is missing {field}.{metric}")
        output[metric] = (
            sum(values) if metric == "invalid_sequence_count" else fmean(values)
        )
    return output


def _precision_from_system(system: dict[str, Any]) -> dict[str, Any]:
    precision = system.get("precision_policy")
    if not isinstance(precision, dict):
        raise ValueError("system record has no precision evidence")
    return dict(precision)


def _final_host_record(
    system: dict[str, Any],
    *,
    config: dict[str, Any],
    config_sha256: str,
    trained_checkpoint_sha256: str,
    application: str,
) -> dict[str, Any]:
    system_acceptance = system.get("acceptance")
    if (
        not isinstance(system_acceptance, dict)
        or system_acceptance.get("system") != "pass"
    ):
        raise ValueError("component system record did not pass")
    training = system.get("training")
    generation = system.get("generation")
    if not isinstance(training, dict) or not isinstance(generation, dict):
        raise ValueError(
            "component system record lacks training or generation evidence"
        )
    return {
        "schema": RECORD_SCHEMA,
        "version": "1.0",
        "source_revision": system.get("source_revision"),
        "flagquantum_version": system.get("flagquantum_version"),
        "kaiwu_pytorch_plugin_revision": system.get("kaiwu_pytorch_plugin_revision"),
        "source_preflight_sha256": system.get("source_preflight_sha256"),
        "transfer_manifest_sha256": system.get("transfer_manifest_sha256"),
        "environment_lock_sha256": system.get("environment_lock_sha256"),
        "python_version": system.get("python_version"),
        "torch_version": system.get("torch_version"),
        "kaiwu_sdk_version": system.get("kaiwu_sdk_version"),
        "experiment_config_sha256": config_sha256,
        "execution_host": system.get("execution_host"),
        "run_role": system.get("run_role"),
        "requested_cuda_device": system.get("requested_cuda_device"),
        "observed_tensor_device": system.get("observed_tensor_device"),
        "observed_gpu_model": system.get("observed_gpu_model"),
        "transport": system.get("transport"),
        "qboson_hardware_used": system.get("qboson_hardware_used"),
        "real_provider_evidence": system.get("real_provider_evidence"),
        "provider_reported_target": system.get("provider_reported_target"),
        "qboson_target": system.get("qboson_target"),
        "qboson_task_ids": system.get("qboson_task_ids"),
        "sampling_mode": system.get("sampling_mode"),
        "requested_samples": system.get("requested_samples"),
        "returned_samples": system.get("returned_samples"),
        "remote_call_budget": system.get("remote_call_budget"),
        "remote_call_count": system.get("remote_call_count"),
        "fallback_occurred": system.get("fallback_occurred"),
        "retrieval_resubmitted": system.get("retrieval_resubmitted"),
        "secrets_redacted": system.get("secrets_redacted"),
        "artifacts": {
            "dataset_sha256": config["dataset"]["sha256"],
            "base_checkpoint_sha256": config["checkpoint"]["sha256"],
            "tokenizer_sha256": config["tokenizer"]["sha256"],
            "evaluation_model_sha256": config["evaluation_model"]["sha256"],
            "trained_energy_checkpoint_sha256": trained_checkpoint_sha256,
        },
        "precision_policy": _precision_from_system(system),
        "training": {
            "energy_objective": training.get("objective"),
            "gradient_norm": training.get("gradient_norm"),
            "parameter_delta_max": training.get("parameter_delta_max"),
        },
        "generation": {
            "token_constraints_passed": generation.get("token_constraints_passed"),
            "invalid_sequence_count": 0,
        },
        "transfer_accounting": system.get("transfer_accounting"),
        "acceptance": {"system": "pass", "application": application},
        "component_bundle_required": True,
    }


def assemble_records(
    *,
    config: dict[str, Any],
    config_sha256: str,
    primary_system: tuple[dict[str, Any], str],
    replay_system: tuple[dict[str, Any], str],
    primary_source_preflight: tuple[dict[str, Any], str],
    replay_source_preflight: tuple[dict[str, Any], str],
    transfer_manifest: tuple[dict[str, Any], str],
    portability: tuple[dict[str, Any], str],
    training_records: list[tuple[dict[str, Any], str]],
    evaluation_records: list[tuple[dict[str, Any], str]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    primary_system_record, primary_system_sha = primary_system
    replay_system_record, replay_system_sha = replay_system
    primary_preflight_record, primary_preflight_sha = primary_source_preflight
    replay_preflight_record, replay_preflight_sha = replay_source_preflight
    transfer_manifest_record, transfer_manifest_sha = transfer_manifest
    portability_record, portability_sha = portability
    if primary_system_record.get("execution_host") != config["primary_host"]:
        raise ValueError("primary system record is from the wrong host")
    if replay_system_record.get("execution_host") != config["replay_host"]:
        raise ValueError("replay system record is from the wrong host")
    if portability_record.get("execution_host") != config["replay_host"]:
        raise ValueError("portability record is from the wrong host")
    if portability_record.get("acceptance") != {"portability": "pass"}:
        raise ValueError("portability replay did not pass")
    for system, label in (
        (primary_system_record, "primary system"),
        (replay_system_record, "replay system"),
    ):
        provider_errors: list[str] = []
        _validate_remote_sampling_component_evidence(system, label, provider_errors)
        if provider_errors:
            raise ValueError("; ".join(provider_errors))
    software = config["software"]
    for preflight, digest, system, host in (
        (
            primary_preflight_record,
            primary_preflight_sha,
            primary_system_record,
            config["primary_host"],
        ),
        (
            replay_preflight_record,
            replay_preflight_sha,
            replay_system_record,
            config["replay_host"],
        ),
    ):
        validate_source_preflight_record(
            preflight,
            execution_host=host,
            source_revision=software["source_revision"],
            plugin_revision=software["kaiwu_pytorch_plugin_revision"],
        )
        if system.get("source_preflight_sha256") != digest:
            raise ValueError(f"system record source preflight differs for {host}")
        if system.get("transfer_manifest_sha256") != preflight.get("manifest_sha256"):
            raise ValueError(f"system record transfer manifest differs for {host}")
    common_manifest_sha = validate_common_transfer_manifest(
        (primary_preflight_record, replay_preflight_record)
    )
    validate_transfer_manifest_record(
        transfer_manifest_record,
        source_revision=software["source_revision"],
        plugin_revision=software["kaiwu_pytorch_plugin_revision"],
    )
    if transfer_manifest_sha != common_manifest_sha:
        raise ValueError("copied transfer manifest differs from source preflights")

    training_by_seed: dict[int, tuple[dict[str, Any], str]] = {}
    for record, digest in training_records:
        seed = record.get("seed")
        if type(seed) is not int or seed in training_by_seed:
            raise ValueError("training records contain an invalid or duplicate seed")
        if record.get("run_completed") is not True:
            raise ValueError(f"training seed {seed} did not complete")
        if record.get("execution_host") != config["primary_host"]:
            raise ValueError(f"training seed {seed} is from the wrong host")
        provider_errors: list[str] = []
        _validate_training_component(
            record,
            config=config,
            config_sha256=config_sha256,
            label=f"training seed {seed}",
            errors=provider_errors,
        )
        if provider_errors:
            raise ValueError("; ".join(provider_errors))
        for field in (
            "source_preflight_sha256",
            "transfer_manifest_sha256",
            "environment_lock_sha256",
        ):
            if record.get(field) != primary_system_record.get(field):
                raise ValueError(f"training seed {seed} has different {field}")
        training_by_seed[seed] = (record, digest)
    if list(sorted(training_by_seed)) != list(sorted(config["seeds"])):
        raise ValueError("training records do not cover the frozen seed set")

    evaluation_by_seed: dict[int, tuple[dict[str, Any], str]] = {}
    for record, digest in evaluation_records:
        seed = record.get("seed")
        if type(seed) is not int or seed in evaluation_by_seed:
            raise ValueError("evaluation records contain an invalid or duplicate seed")
        training_entry = training_by_seed.get(seed)
        if (
            training_entry is None
            or record.get("training_record_sha256") != training_entry[1]
        ):
            raise ValueError(
                f"evaluation seed {seed} is not linked to its training record"
            )
        if record.get("experiment_config_sha256") != config_sha256:
            raise ValueError(f"evaluation seed {seed} uses another frozen config")
        if record.get("execution_host") != config["primary_host"]:
            raise ValueError(f"evaluation seed {seed} is from the wrong host")
        evaluation_errors: list[str] = []
        _validate_evaluation_component(
            record,
            config=config,
            config_sha256=config_sha256,
            label=f"evaluation seed {seed}",
            errors=evaluation_errors,
        )
        if evaluation_errors:
            raise ValueError("; ".join(evaluation_errors))
        for field in (
            "source_preflight_sha256",
            "transfer_manifest_sha256",
            "environment_lock_sha256",
        ):
            if record.get(field) != primary_system_record.get(field):
                raise ValueError(f"evaluation seed {seed} has different {field}")
        evaluation_by_seed[seed] = (record, digest)
    if set(evaluation_by_seed) != set(training_by_seed):
        raise ValueError("evaluation records do not cover every training seed")

    portability_seed = config["generation"]["portability_training_seed"]
    portability_training, portability_training_sha = training_by_seed[portability_seed]
    checkpoint_digest = portability_training.get("trained_energy_checkpoint_sha256")
    if not isinstance(checkpoint_digest, str) or len(checkpoint_digest) != 64:
        raise ValueError("selected training record has no checkpoint digest")
    if portability_record.get("training_record_sha256") != portability_training_sha:
        raise ValueError("portability replay is linked to another training record")
    portability_errors: list[str] = []
    _validate_portability_component_evidence(
        portability_record,
        "portability replay",
        portability_errors,
        expected_requested_samples=config.get("requested_samples"),
        config=config,
    )
    if portability_errors:
        raise ValueError("; ".join(portability_errors))
    if portability_record.get("trained_energy_checkpoint_sha256") != checkpoint_digest:
        raise ValueError("portability replay used another trained checkpoint")
    for field in (
        "source_preflight_sha256",
        "transfer_manifest_sha256",
        "environment_lock_sha256",
    ):
        if portability_record.get(field) != replay_system_record.get(field):
            raise ValueError(f"portability replay has different {field}")

    evaluations = [evaluation_by_seed[seed][0] for seed in config["seeds"]]
    primary = _final_host_record(
        primary_system_record,
        config=config,
        config_sha256=config_sha256,
        trained_checkpoint_sha256=checkpoint_digest,
        application="pass",
    )
    primary["attempted_seeds"] = list(config["seeds"])
    primary["baseline_metrics"] = _aggregate_metrics(evaluations, "baseline_metrics")
    primary["guided_metrics"] = _aggregate_metrics(evaluations, "guided_metrics")
    primary["system_evidence_sha256"] = primary_system_sha
    primary["application_evidence"] = {
        "aggregation": "arithmetic_mean_across_frozen_seeds",
        "records": [
            {
                "seed": seed,
                "training_record_sha256": training_by_seed[seed][1],
                "evaluation_record_sha256": evaluation_by_seed[seed][1],
                "trained_energy_checkpoint_sha256": training_by_seed[seed][0].get(
                    "trained_energy_checkpoint_sha256"
                ),
            }
            for seed in config["seeds"]
        ],
    }
    replay = _final_host_record(
        replay_system_record,
        config=config,
        config_sha256=config_sha256,
        trained_checkpoint_sha256=checkpoint_digest,
        application="not_run",
    )
    replay["system_evidence_sha256"] = replay_system_sha
    replay["portability_evidence"] = {
        "record_sha256": portability_sha,
        "training_seed": portability_seed,
        "training_record_sha256": portability_training_sha,
        "trained_energy_checkpoint_sha256": checkpoint_digest,
        "acceptance": "pass",
    }

    errors: list[str] = []
    _validate_system_record(
        primary,
        config=config,
        config_sha256=config_sha256,
        label="primary",
        errors=errors,
    )
    _validate_system_record(
        replay,
        config=config,
        config_sha256=config_sha256,
        label="replay",
        errors=errors,
    )
    _validate_application(primary, config, errors)
    if errors:
        raise ValueError("assembled records failed validation: " + "; ".join(errors))
    return primary, replay


def _write_exclusive(path: Path, encoded: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(encoded)


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()


def _materialize_acceptance_bundle(
    root: Path,
    *,
    config_bytes: bytes,
    environment_lock_bytes: bytes,
    config: dict[str, Any],
    primary: dict[str, Any],
    replay: dict[str, Any],
    component_bytes: dict[str, bytes],
) -> Path:
    """Write a complete candidate bundle inside an existing private directory."""

    components = root / "components"
    components.mkdir(mode=0o700)
    _write_exclusive(root / "acceptance_config.json", config_bytes)
    _write_exclusive(root / "environment_lock.json", environment_lock_bytes)
    component_entries = []
    for name, encoded in component_bytes.items():
        destination = components / name
        _write_exclusive(destination, encoded)
        component_entries.append(
            {
                "path": f"components/{name}",
                "sha256": hashlib.sha256(encoded).hexdigest(),
            }
        )

    primary_name = f"{config['primary_host']}.json"
    replay_name = f"{config['replay_host']}.json"
    primary_bytes = _json_bytes(primary)
    replay_bytes = _json_bytes(replay)
    _write_exclusive(root / primary_name, primary_bytes)
    _write_exclusive(root / replay_name, replay_bytes)
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "version": "1.0",
        "config": {
            "path": "acceptance_config.json",
            "sha256": hashlib.sha256(config_bytes).hexdigest(),
        },
        "environment_lock": {
            "path": "environment_lock.json",
            "sha256": hashlib.sha256(environment_lock_bytes).hexdigest(),
        },
        "records": [
            {"path": primary_name, "sha256": hashlib.sha256(primary_bytes).hexdigest()},
            {"path": replay_name, "sha256": hashlib.sha256(replay_bytes).hexdigest()},
        ],
        "component_records": component_entries,
    }
    manifest_path = root / "manifest.json"
    _write_exclusive(manifest_path, _json_bytes(manifest))
    return manifest_path


def _publish_acceptance_bundle(
    destination: Path,
    *,
    config_path: Path,
    environment_lock_path: Path,
    config: dict[str, Any],
    primary: dict[str, Any],
    replay: dict[str, Any],
    component_sources: dict[str, Path],
    expected_config_sha256: str | None = None,
    expected_component_sha256: dict[str, str] | None = None,
) -> None:
    """Validate in a private sibling directory before atomically publishing."""

    config_bytes = read_private_bytes(
        config_path,
        label="frozen configuration",
        max_bytes=_MAX_CONFIG_BYTES,
    )
    config_digest = hashlib.sha256(config_bytes).hexdigest()
    if expected_config_sha256 is not None and config_digest != expected_config_sha256:
        raise ValueError("frozen configuration changed after assembly")
    environment_lock_bytes = read_private_bytes(
        environment_lock_path,
        label="environment lock",
        max_bytes=_MAX_ENVIRONMENT_LOCK_BYTES,
    )
    if (
        expected_component_sha256 is not None
        and expected_component_sha256.keys() != component_sources.keys()
    ):
        raise ValueError("component digest set differs from component sources")
    component_bytes: dict[str, bytes] = {}
    for name, source in component_sources.items():
        encoded = read_private_bytes(
            source,
            label=f"component record {name}",
            max_bytes=_MAX_COMPONENT_BYTES,
        )
        digest = hashlib.sha256(encoded).hexdigest()
        if (
            expected_component_sha256 is not None
            and expected_component_sha256.get(name) != digest
        ):
            raise ValueError(f"component record changed after assembly: {name}")
        component_bytes[name] = encoded
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    _require_private_directory(
        destination.parent, description="acceptance output parent"
    )
    with TemporaryDirectory(
        prefix=f".{destination.name}.staging-",
        dir=destination.parent,
    ) as temporary_directory:
        staging = Path(temporary_directory)
        staging.chmod(0o700)
        manifest_path = _materialize_acceptance_bundle(
            staging,
            config_bytes=config_bytes,
            environment_lock_bytes=environment_lock_bytes,
            config=config,
            primary=primary,
            replay=replay,
            component_bytes=component_bytes,
        )
        errors = validate_acceptance(manifest_path)
        if errors:
            raise RuntimeError(
                "final acceptance validation failed: " + "; ".join(errors)
            )
        if destination.exists():
            raise RuntimeError("--evidence-dir appeared during final validation")
        os.replace(staging, destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--environment-lock", required=True, type=Path)
    parser.add_argument("--primary-system", required=True, type=Path)
    parser.add_argument("--replay-system", required=True, type=Path)
    parser.add_argument("--primary-source-preflight", required=True, type=Path)
    parser.add_argument("--replay-source-preflight", required=True, type=Path)
    parser.add_argument("--transfer-manifest", required=True, type=Path)
    parser.add_argument("--sdk-approval", required=True, type=Path)
    parser.add_argument("--provider-smoke", required=True, type=Path)
    parser.add_argument("--artifact-preflight", required=True, type=Path)
    parser.add_argument("--portability", required=True, type=Path)
    parser.add_argument("--training-record", action="append", required=True, type=Path)
    parser.add_argument(
        "--evaluation-record", action="append", required=True, type=Path
    )
    parser.add_argument("--evidence-dir", required=True, type=Path)
    args = parser.parse_args()
    input_paths = [
        args.config,
        args.environment_lock,
        args.primary_system,
        args.replay_system,
        args.primary_source_preflight,
        args.replay_source_preflight,
        args.transfer_manifest,
        args.sdk_approval,
        args.provider_smoke,
        args.artifact_preflight,
        args.portability,
        *args.training_record,
        *args.evaluation_record,
    ]
    if any(not path.is_absolute() for path in (*input_paths, args.evidence_dir)):
        parser.error("every path must be absolute")
    if args.evidence_dir.exists():
        parser.error("--evidence-dir must not already exist")
    try:
        for path in input_paths:
            _require_private_input(path)
    except ValueError as exc:
        parser.error(str(exc))

    config, config_sha256 = _load_frozen_config(args.config)
    primary_system = _load_component(args.primary_system, SYSTEM_SCHEMA)
    replay_system = _load_component(args.replay_system, SYSTEM_SCHEMA)
    primary_source_preflight = _load_component(
        args.primary_source_preflight, SOURCE_PREFLIGHT_SCHEMA
    )
    replay_source_preflight = _load_component(
        args.replay_source_preflight, SOURCE_PREFLIGHT_SCHEMA
    )
    transfer_manifest = _load_component(
        args.transfer_manifest, TRANSFER_MANIFEST_SCHEMA
    )
    sdk_approval = _load_component(args.sdk_approval, SDK_APPROVAL_SCHEMA)
    provider_smoke = _load_component(args.provider_smoke, PROVIDER_SMOKE_SCHEMA)
    artifact_preflight = _load_component(
        args.artifact_preflight, ARTIFACT_PREFLIGHT_SCHEMA
    )
    portability = _load_component(args.portability, PORTABILITY_SCHEMA)
    training = [_load_component(path, TRAINING_SCHEMA) for path in args.training_record]
    evaluations = [
        _load_component(path, EVALUATION_SCHEMA) for path in args.evaluation_record
    ]
    primary, replay = assemble_records(
        config=config,
        config_sha256=config_sha256,
        primary_system=primary_system,
        replay_system=replay_system,
        primary_source_preflight=primary_source_preflight,
        replay_source_preflight=replay_source_preflight,
        transfer_manifest=transfer_manifest,
        portability=portability,
        training_records=training,
        evaluation_records=evaluations,
    )

    component_sources = {
        "primary-system.json": args.primary_system,
        "replay-system.json": args.replay_system,
        "primary-source-preflight.json": args.primary_source_preflight,
        "replay-source-preflight.json": args.replay_source_preflight,
        "transfer-manifest.json": args.transfer_manifest,
        "sdk-approval.json": args.sdk_approval,
        "provider-smoke.json": args.provider_smoke,
        "artifact-preflight.json": args.artifact_preflight,
        "portability.json": args.portability,
    }
    for index, path in enumerate(args.training_record):
        component_sources[f"training-{index}.json"] = path
    for index, path in enumerate(args.evaluation_record):
        component_sources[f"evaluation-{index}.json"] = path
    component_digests = {
        "primary-system.json": primary_system[1],
        "replay-system.json": replay_system[1],
        "primary-source-preflight.json": primary_source_preflight[1],
        "replay-source-preflight.json": replay_source_preflight[1],
        "transfer-manifest.json": transfer_manifest[1],
        "sdk-approval.json": sdk_approval[1],
        "provider-smoke.json": provider_smoke[1],
        "artifact-preflight.json": artifact_preflight[1],
        "portability.json": portability[1],
    }
    for index, (_, digest) in enumerate(training):
        component_digests[f"training-{index}.json"] = digest
    for index, (_, digest) in enumerate(evaluations):
        component_digests[f"evaluation-{index}.json"] = digest
    _publish_acceptance_bundle(
        args.evidence_dir,
        config_path=args.config,
        environment_lock_path=args.environment_lock,
        config=config,
        primary=primary,
        replay=replay,
        component_sources=component_sources,
        expected_config_sha256=config_sha256,
        expected_component_sha256=component_digests,
    )
    print(f"Validated acceptance bundle written to {args.evidence_dir}")


if __name__ == "__main__":
    main()
