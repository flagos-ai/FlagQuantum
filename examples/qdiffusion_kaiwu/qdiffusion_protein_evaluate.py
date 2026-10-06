"""Evaluate frozen QDiffusion outputs with one local ESM2 checkpoint on A800."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import platform
import re
import socket
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any

import torch

from examples.qdiffusion_kaiwu.preflight_protein_artifacts import (
    AMINO_ACIDS,
    _artifact_identity,
)
from examples.qdiffusion_kaiwu.private_io import (
    read_private_bytes,
    validate_private_json_output_path,
)
from examples.qdiffusion_kaiwu.qdiffusion_system_development_probe import (
    _validate_imported_module_tree,
)
from examples.qdiffusion_kaiwu.qdiffusion_system_live import (
    _load_frozen_config,
    _validate_lane,
    _write_private_redacted_json,
)
from examples.qdiffusion_kaiwu.sdk_approval import (
    verify_approved_kaiwu_distribution,
)
from examples.qdiffusion_kaiwu.source_preflight import load_source_preflight
from examples.qdiffusion_kaiwu.stable_source_tree import (
    RegularFileSnapshot,
    capture_regular_file,
    open_captured_regular_file,
    revalidate_regular_file,
)
from examples.qdiffusion_kaiwu.strict_json import loads_json_strict
from examples.qdiffusion_kaiwu.validate_acceptance import (
    _validate_training_provider_evidence,
)
from examples.qdiffusion_kaiwu.verify_environment_lock import (
    verify_frozen_environment_lock,
)
from flagquantum.version import __version__ as flagquantum_version

SCHEMA = "flagquantum.qboson_qdiffusion_protein_evaluation"
TRAINING_SCHEMA = "flagquantum.qboson_qdiffusion_protein_training"
_MAX_TRAINING_RECORD_BYTES = 64 * 1024 * 1024


class _VerifiedTrainingPaths(dict[str, Path]):
    """Training paths paired with the exact file identities already accepted."""

    def __init__(
        self, paths: dict[str, Path], snapshots: dict[str, RegularFileSnapshot]
    ) -> None:
        super().__init__(paths)
        self.snapshots = snapshots


def _load_pinned_eval_workflow(plugin_root: Path) -> tuple[ModuleType, ModuleType]:
    root = plugin_root.resolve()
    case_root = root / "example" / "qdiffusion"
    source_root = root / "src"
    expected = case_root / "dplm" / "workflows" / "esm2_eval.py"
    if not expected.is_file() or not source_root.is_dir():
        raise ValueError("plugin root does not contain the pinned ESM2 workflow")
    for path in (source_root, case_root):
        encoded = str(path)
        if encoded not in sys.path:
            sys.path.insert(0, encoded)
    workflow = importlib.import_module("dplm.workflows.esm2_eval")
    helpers = importlib.import_module("dplm.workflows.esm2_eval_helpers")
    if Path(str(workflow.__file__)).resolve() != expected.resolve():
        raise RuntimeError("imported ESM2 workflow is outside --plugin-root")
    expected_helpers = expected.with_name("esm2_eval_helpers.py").resolve()
    if Path(str(helpers.__file__)).resolve() != expected_helpers:
        raise RuntimeError("imported ESM2 helpers are outside --plugin-root")
    _validate_imported_module_tree(
        module_prefix="dplm",
        expected_root=case_root / "dplm",
        label="DPLM",
    )
    return workflow, helpers


def _load_training_record(path: Path) -> tuple[dict[str, Any], str]:
    encoded = read_private_bytes(
        path,
        label="protein-training record",
        max_bytes=_MAX_TRAINING_RECORD_BYTES,
    )
    record = loads_json_strict(encoded)
    if not isinstance(record, dict):
        raise ValueError("training record must be a JSON object")
    if record.get("schema") != TRAINING_SCHEMA or record.get("version") != "1.0":
        raise ValueError("unsupported protein-training record")
    if record.get("run_completed") is not True:
        raise ValueError("protein-training record is not complete")
    provider_errors: list[str] = []
    _validate_training_provider_evidence(
        record, "protein-training record", provider_errors
    )
    if provider_errors:
        raise ValueError(
            "protein-training provider evidence is invalid: "
            + "; ".join(provider_errors)
        )
    return record, hashlib.sha256(encoded).hexdigest()


def _verified_training_paths(
    run_directory: Path,
    record: dict[str, Any],
) -> _VerifiedTrainingPaths:
    if run_directory.name != record.get("run_directory_name"):
        raise ValueError("training run directory name differs from its record")
    raw_artifacts = record.get("workflow_artifacts")
    if not isinstance(raw_artifacts, dict):
        raise ValueError("training record has no workflow artifact identities")
    required = (
        "test_fasta",
        "baseline_fasta",
        "guided_fasta",
        "baseline_quality",
        "guided_quality",
    )
    paths: dict[str, Path] = {}
    snapshots: dict[str, RegularFileSnapshot] = {}
    for name in required:
        identity = raw_artifacts.get(name)
        if not isinstance(identity, dict):
            raise ValueError(f"training record is missing {name}")
        relative = identity.get("relative_path")
        expected = identity.get("sha256")
        if not isinstance(relative, str) or not isinstance(expected, str):
            raise ValueError(f"training record has an invalid {name} identity")
        candidate = run_directory / relative
        try:
            candidate.resolve().relative_to(run_directory.resolve())
        except ValueError:
            raise ValueError(
                f"training record {name} escapes the run directory"
            ) from None
        try:
            snapshot = capture_regular_file(
                candidate, label=f"training artifact {name}"
            )
        except ValueError:
            raise ValueError(
                f"training artifact is absent or symbolic: {name}"
            ) from None
        if snapshot.sha256 != expected:
            raise ValueError(f"training artifact digest mismatch: {name}")
        paths[name] = candidate
        snapshots[name] = snapshot
    return _VerifiedTrainingPaths(paths, snapshots)


def _revalidate_training_paths(paths: dict[str, Path]) -> None:
    if not isinstance(paths, _VerifiedTrainingPaths):
        return
    for name, snapshot in paths.snapshots.items():
        revalidate_regular_file(snapshot, label=f"training artifact {name}")


def _bind_training_paths(paths: dict[str, Path]) -> _VerifiedTrainingPaths:
    if isinstance(paths, _VerifiedTrainingPaths):
        return paths
    snapshots = {
        name: capture_regular_file(path, label=f"training artifact {name}")
        for name, path in paths.items()
    }
    return _VerifiedTrainingPaths(dict(paths), snapshots)


def _read_aligned_records(
    workflow: Any,
    paths: dict[str, Path],
    *,
    expected_count: int,
) -> tuple[
    list[tuple[str, str]],
    list[tuple[str, str]],
    list[tuple[str, str]],
]:
    paths = _bind_training_paths(paths)
    _revalidate_training_paths(paths)
    groups: list[list[tuple[str, str]]] = []
    for name in ("test_fasta", "baseline_fasta", "guided_fasta"):
        records = [
            (header, workflow.normalize_sequence(sequence))
            for header, sequence in workflow.read_fasta_records(paths[name])
        ]
        groups.append(records)
    _revalidate_training_paths(paths)
    if any(len(records) != expected_count for records in groups):
        raise ValueError("frozen FASTA counts differ from generation.sequence_count")
    headers = [[header for header, _ in records] for records in groups]
    if headers[0] != headers[1] or headers[0] != headers[2]:
        raise ValueError(
            "reference, baseline, and guided FASTA headers are not aligned"
        )
    return groups[0], groups[1], groups[2]


def _invalid_sequence_count(records: list[tuple[str, str]]) -> int:
    return sum(
        not sequence or bool(set(sequence.upper()) - AMINO_ACIDS)
        for _, sequence in records
    )


def _source_preflight_identity(record: dict[str, Any]) -> tuple[str, str]:
    values: list[str] = []
    for field in ("source_preflight_sha256", "transfer_manifest_sha256"):
        value = record.get(field)
        if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise ValueError(f"training record has invalid {field}")
        values.append(value)
    return values[0], values[1]


def _verified_evaluation_source(
    path: Path,
    training_record: dict[str, Any],
    *,
    execution_host: str,
    source_revision: str,
    plugin_revision: str,
    source_root: Path,
    plugin_root: Path,
) -> tuple[str, str]:
    expected_preflight_sha256, expected_manifest_sha256 = _source_preflight_identity(
        training_record
    )
    source_preflight, source_preflight_sha256 = load_source_preflight(
        path,
        execution_host=execution_host,
        source_revision=source_revision,
        plugin_revision=plugin_revision,
        source_root=source_root,
        plugin_root=plugin_root,
    )
    transfer_manifest_sha256 = source_preflight["manifest_sha256"]
    if source_preflight_sha256 != expected_preflight_sha256:
        raise ValueError("evaluation source preflight differs from training record")
    if transfer_manifest_sha256 != expected_manifest_sha256:
        raise ValueError("evaluation transfer manifest differs from training record")
    return source_preflight_sha256, transfer_manifest_sha256


def _local_esm2_model(
    helpers: Any,
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[Any, Any]:
    pretrained = getattr(getattr(helpers, "esm", None), "pretrained", None)
    loader = getattr(pretrained, "load_model_and_alphabet_local", None)
    if not callable(loader):
        raise RuntimeError("ESM does not expose load_model_and_alphabet_local")
    model, alphabet = loader(str(checkpoint_path))
    return model.eval().to(device), alphabet


def _quality_metrics(
    path: Path, snapshot: RegularFileSnapshot | None = None
) -> dict[str, Any]:
    if snapshot is None:
        snapshot = capture_regular_file(path, label="training quality artifact")
    with open_captured_regular_file(
        snapshot, label="training quality artifact"
    ) as stream:
        encoded = stream.read()
    value = loads_json_strict(encoded)
    if not isinstance(value, dict):
        raise ValueError("sequence quality summary must be a JSON object")
    return value


def evaluate_outputs(
    *,
    workflow: Any,
    helpers: Any,
    paths: dict[str, Path],
    evaluation_checkpoint: Path,
    config: dict[str, Any],
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any]]:
    paths = _bind_training_paths(paths)
    reference, baseline, guided = _read_aligned_records(
        workflow,
        paths,
        expected_count=config["generation"]["sequence_count"],
    )
    model, alphabet = _local_esm2_model(helpers, evaluation_checkpoint, device)
    evaluation = config["evaluation"]
    reference_embeddings = workflow.embed_sequences(
        reference,
        model=model,
        alphabet=alphabet,
        device=device,
        batch_size=evaluation["batch_size"],
        pooling=evaluation["pooling"],
    )

    results: list[dict[str, Any]] = []
    for label, records, quality_name in (
        ("baseline", baseline, "baseline_quality"),
        ("guided", guided, "guided_quality"),
    ):
        embeddings = workflow.embed_sequences(
            records,
            model=model,
            alphabet=alphabet,
            device=device,
            batch_size=evaluation["batch_size"],
            pooling=evaluation["pooling"],
        )
        _, summary = workflow.evaluate_candidate_set(
            label=label,
            reference_records=reference,
            candidate_records=records,
            reference_embeddings=reference_embeddings,
            candidate_embeddings=embeddings,
            pair_mode=evaluation["pair_mode"],
        )
        metrics = asdict(summary)
        quality_snapshot = (
            paths.snapshots[quality_name]
            if isinstance(paths, _VerifiedTrainingPaths)
            else None
        )
        quality = _quality_metrics(paths[quality_name], quality_snapshot)
        for field in (
            "identity_to_reference_mean",
            "amino_acid_jsd",
            "kmer2_jsd",
            "kmer3_jsd",
            "uniqueness_ratio",
            "repeat_ratio_ge4",
            "length_match_ratio",
        ):
            metrics[field] = quality.get(field)
        metrics["invalid_sequence_count"] = _invalid_sequence_count(records)
        results.append(metrics)
    _revalidate_training_paths(paths)
    return results[0], results[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--training-record", required=True, type=Path)
    parser.add_argument("--run-directory", required=True, type=Path)
    parser.add_argument("--plugin-root", required=True, type=Path)
    parser.add_argument("--evaluation-model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--execution-host",
        choices=("jp-a800-171", "jp-a800-172"),
        required=True,
    )
    parser.add_argument("--expected-hostname", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--plugin-revision", required=True)
    parser.add_argument("--source-preflight", required=True, type=Path)
    parser.add_argument("--environment-lock", required=True, type=Path)
    parser.add_argument("--expected-sdk-version", choices=("1.3.1",), default="1.3.1")
    args = parser.parse_args()
    for label, path in {
        "config": args.config,
        "training-record": args.training_record,
        "run-directory": args.run_directory,
        "plugin-root": args.plugin_root,
        "evaluation-model": args.evaluation_model,
        "source-preflight": args.source_preflight,
        "environment-lock": args.environment_lock,
        "output": args.output,
    }.items():
        if not path.is_absolute():
            parser.error(f"--{label} must be an absolute path")
    validate_private_json_output_path(args.output)

    config, config_sha256 = _load_frozen_config(args.config)
    record, record_sha256 = _load_training_record(args.training_record)
    if record.get("experiment_config_sha256") != config_sha256:
        parser.error("training record belongs to another frozen configuration")
    if record.get("source_revision") != args.source_revision:
        parser.error("training source revision differs from --source-revision")
    if record.get("kaiwu_pytorch_plugin_revision") != args.plugin_revision:
        parser.error("training plugin revision differs from --plugin-revision")
    try:
        source_preflight_sha256, transfer_manifest_sha256 = _verified_evaluation_source(
            args.source_preflight,
            record,
            execution_host=args.execution_host,
            source_revision=args.source_revision,
            plugin_revision=args.plugin_revision,
            source_root=Path(__file__).resolve().parents[2],
            plugin_root=args.plugin_root,
        )
    except ValueError as exc:
        parser.error(str(exc))
    environment_record, environment_lock_sha256 = verify_frozen_environment_lock(
        args.environment_lock,
        expected_sha256=config["software"]["environment_lock_sha256"],
    )
    verify_approved_kaiwu_distribution(environment_record, config["kaiwu_sdk"])
    if args.expected_hostname != config["host_identities"][args.execution_host]:
        parser.error("--expected-hostname differs from the frozen host identity")
    hostname = socket.gethostname()
    if hostname != args.expected_hostname:
        parser.error("observed hostname differs from --expected-hostname")
    role, _, _ = _validate_lane(
        config,
        execution_host=args.execution_host,
        source_revision=args.source_revision,
        plugin_revision=args.plugin_revision,
        sdk_version=args.expected_sdk_version,
    )
    if role != "primary" or record.get("execution_host") != args.execution_host:
        parser.error("ESM2 evaluation must run with its primary-host training record")
    device = torch.device("cuda:0")
    if not torch.cuda.is_available():
        parser.error("ESM2 evaluation requires CUDA")
    torch.cuda.set_device(device)
    gpu = torch.cuda.get_device_name(device)
    if "A800" not in gpu:
        parser.error("ESM2 evaluation requires an NVIDIA A800")

    evaluation_digest, algorithm, file_count = _artifact_identity(args.evaluation_model)
    if algorithm != "file-sha256-v1" or file_count != 1:
        parser.error("ESM2 evaluation model must be one local checkpoint file")
    if evaluation_digest != config["evaluation_model"]["sha256"]:
        parser.error("ESM2 checkpoint digest differs from frozen configuration")
    paths = _verified_training_paths(args.run_directory, record)
    workflow, helpers = _load_pinned_eval_workflow(args.plugin_root)
    baseline, guided = evaluate_outputs(
        workflow=workflow,
        helpers=helpers,
        paths=paths,
        evaluation_checkpoint=args.evaluation_model,
        config=config,
        device=device,
    )
    postflight_digest, postflight_algorithm, postflight_file_count = _artifact_identity(
        args.evaluation_model
    )
    if (
        postflight_digest != evaluation_digest
        or postflight_algorithm != algorithm
        or postflight_file_count != file_count
    ):
        raise RuntimeError("ESM2 checkpoint changed during protein evaluation")
    _revalidate_training_paths(paths)
    payload = {
        "schema": SCHEMA,
        "version": "1.0",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "experiment_config_sha256": config_sha256,
        "training_record_sha256": record_sha256,
        "source_revision": args.source_revision,
        "flagquantum_version": flagquantum_version,
        "kaiwu_pytorch_plugin_revision": args.plugin_revision,
        "source_preflight_sha256": source_preflight_sha256,
        "transfer_manifest_sha256": transfer_manifest_sha256,
        "environment_lock_sha256": environment_lock_sha256,
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "kaiwu_sdk_version": args.expected_sdk_version,
        "execution_host": args.execution_host,
        "observed_hostname": hostname,
        "observed_gpu_model": gpu,
        "observed_tensor_device": str(device),
        "seed": record.get("seed"),
        "evaluation_model_sha256": evaluation_digest,
        "artifact_inputs_unchanged": True,
        "baseline_metrics": baseline,
        "guided_metrics": guided,
        "secrets_redacted": True,
        "provider_quota_consumed": False,
        "acceptance": "candidate_evidence_only",
    }
    _write_private_redacted_json(args.output, payload, forbidden_values=())
    print(f"Private protein-evaluation record written to {args.output}")


if __name__ == "__main__":
    main()
