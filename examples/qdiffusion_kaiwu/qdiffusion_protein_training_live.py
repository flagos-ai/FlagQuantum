"""Run one frozen QDiffusion protein-training seed through FlagQuantum Kaiwu.

This command consumes QBoson quota. It runs only the training workflow; ESM2
evaluation and final application acceptance are separate gates.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import os
import platform
import re
import socket
import stat
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any

import torch

from examples.qdiffusion_kaiwu.failure_evidence import (
    apply_artifact_postflight,
    redacted_failure_record,
)
from examples.qdiffusion_kaiwu.plan_quota import (
    estimate_protein_remote_calls as _estimate_protein_remote_calls,
)
from examples.qdiffusion_kaiwu.preflight_protein_artifacts import (
    ArtifactSnapshot,
    assert_artifacts_unchanged,
    preflight_artifacts_with_snapshots,
    revalidate_artifact_snapshots,
)
from examples.qdiffusion_kaiwu.private_io import (
    read_private_bytes,
    validate_private_directory,
    validate_private_json_output_path,
)
from examples.qdiffusion_kaiwu.provider_inputs import normalize_provider_identifier
from examples.qdiffusion_kaiwu.provider_resources import (
    assess_provider_budget,
    build_provider_resource_gate,
    load_provider_resources,
    provider_resource_valid_until,
    validate_provider_resource_gate,
)
from examples.qdiffusion_kaiwu.qdiffusion_system_development_probe import (
    _load_pinned_qdiffusion_api,
    _validate_imported_module_tree,
)
from examples.qdiffusion_kaiwu.qdiffusion_system_live import (
    ACKNOWLEDGEMENT,
    _completed_provider_result_has_identity,
    _load_frozen_config,
    _precision_evidence_complete,
    _validate_lane,
    _write_private_redacted_json,
)
from examples.qdiffusion_kaiwu.sdk_approval import (
    verify_approved_kaiwu_distribution,
    verify_approved_project_assignment,
)
from examples.qdiffusion_kaiwu.source_preflight import load_source_preflight
from examples.qdiffusion_kaiwu.stable_source_tree import (
    RegularFileSnapshot,
    capture_regular_file,
    revalidate_regular_file,
)
from examples.qdiffusion_kaiwu.verify_environment_lock import (
    verify_frozen_environment_lock,
)
from flagquantum.ecosystem.kaiwu import (
    KaiwuSampler,
    bound_qdiffusion_workflow,
)
from flagquantum.remote.kaiwu import (
    KaiwuCredentials,
    KaiwuSDKClient,
    resolve_kaiwu_credentials,
)
from flagquantum.version import __version__ as flagquantum_version

SCHEMA = "flagquantum.qboson_qdiffusion_protein_training"
_MAX_ARTIFACT_PREFLIGHT_BYTES = 4 * 1024 * 1024


def _effective_uid() -> int:
    getter = getattr(os, "geteuid", None)
    if not callable(getter):
        raise ValueError("platform cannot validate protein output ownership")
    return int(getter())


def _load_pinned_workflow(plugin_root: Path) -> ModuleType:
    root = plugin_root.resolve()
    case_root = root / "example" / "qdiffusion"
    source_root = root / "src"
    expected = case_root / "dplm" / "workflows" / "train.py"
    if not expected.is_file() or not source_root.is_dir():
        raise ValueError("plugin root does not contain the pinned QDiffusion workflow")
    _load_pinned_qdiffusion_api(root)
    for path in (source_root, case_root):
        encoded = str(path)
        if encoded not in sys.path:
            sys.path.insert(0, encoded)
    module = importlib.import_module("dplm.workflows.train")
    module_path = Path(str(module.__file__)).resolve()
    if module_path != expected.resolve():
        raise RuntimeError("imported QDiffusion workflow is outside --plugin-root")
    _validate_imported_module_tree(
        module_prefix="dplm",
        expected_root=case_root / "dplm",
        label="DPLM",
    )
    return module


def _build_workflow_config(
    workflow: Any,
    config: dict[str, Any],
    *,
    dataset_path: Path,
    checkpoint_path: Path,
    seed: int,
) -> Any:
    dataset = config["dataset"]
    training = config["training"]
    generation = config["generation"]
    return workflow.WorkflowConfig(
        data=workflow.DataConfig(
            fasta_path=str(dataset_path),
            min_length=dataset["min_length"],
            max_length=dataset["max_length"],
            max_records=dataset["max_records"],
            val_ratio=dataset["validation_ratio"],
            test_ratio=dataset["test_ratio"],
            seed=seed,
        ),
        model=workflow.ModelConfig(
            proposal_ckpt=str(checkpoint_path),
            energy_ckpt=str(checkpoint_path),
            freeze_proposal=training["freeze_proposal"],
        ),
        sampler=workflow.SamplerConfig(
            sampler_type="flagquantum-kaiwu",
            sampler_kwargs={},
        ),
        train=workflow.TrainConfig(
            epochs=training["epochs"],
            min_epochs=training["min_epochs"],
            batch_size=training["batch_size"],
            learning_rate=training["learning_rate"],
            weight_decay=training["weight_decay"],
            grad_clip_norm=training["grad_clip_norm"],
            num_candidates=training["num_candidates"],
            validation_steps=training["validation_steps"],
            scheduler_factor=training["scheduler_factor"],
            scheduler_patience=training["scheduler_patience"],
            early_stop_patience=training["early_stop_patience"],
            require_cuda=True,
        ),
        generate=workflow.GenerateConfig(
            num_candidates=generation["num_candidates"],
            energy_temperature=generation["energy_temperature"],
            proposal_temperature=generation["proposal_temperature"],
            proposal_noise_scale=generation["proposal_noise_scale"],
            disable_resample=generation["disable_resample"],
            resample_ratio=generation["resample_ratio"],
            resample_top_p=generation["resample_top_p"],
            steps=generation["max_steps"],
        ),
    )


def _run_workflow(
    workflow: Any,
    workflow_config: Any,
    sampler: KaiwuSampler,
    output_root: Path,
) -> Path:
    original_umask = os.umask(0o077)
    try:
        output_root.mkdir(mode=0o700, parents=True, exist_ok=True)
        output_metadata = output_root.lstat()
        if (
            output_root.is_symlink()
            or not stat.S_ISDIR(output_metadata.st_mode)
            or stat.S_IMODE(output_metadata.st_mode) & 0o077
            or output_metadata.st_uid != _effective_uid()
        ):
            raise ValueError(
                "protein workflow output root must be a private real directory"
            )
        before = {item.name for item in output_root.iterdir()}
        original_config_factory = workflow.build_default_workflow_config
        original_outputs_factory = workflow.default_outputs_root
        workflow.build_default_workflow_config = lambda: workflow_config
        workflow.default_outputs_root = lambda: output_root
        try:
            with bound_qdiffusion_workflow(workflow, sampler):
                workflow.main()
        finally:
            workflow.build_default_workflow_config = original_config_factory
            workflow.default_outputs_root = original_outputs_factory
        created = [item for item in output_root.iterdir() if item.name not in before]
        created_metadata = created[0].lstat() if len(created) == 1 else None
        if (
            len(created) != 1
            or created[0].is_symlink()
            or created_metadata is None
            or not stat.S_ISDIR(created_metadata.st_mode)
            or stat.S_IMODE(created_metadata.st_mode) & 0o077
            or created_metadata.st_uid != _effective_uid()
        ):
            raise RuntimeError(
                "protein workflow did not create exactly one private run directory"
            )
        return created[0]
    finally:
        os.umask(original_umask)


def _capture_private_training_output(
    path: Path, *, run_directory: Path, label: str
) -> RegularFileSnapshot:
    try:
        path.relative_to(run_directory)
    except ValueError:
        raise ValueError(f"{label} is outside the training run directory") from None
    snapshot = capture_regular_file(path, label=label)
    metadata = path.lstat()
    if stat.S_IMODE(metadata.st_mode) & 0o077:
        raise ValueError(f"{label} must be owner-only")
    if metadata.st_uid != _effective_uid():
        raise ValueError(f"{label} must be owned by the current effective user")
    return snapshot


def _checkpoint_snapshot(run_directory: Path) -> RegularFileSnapshot:
    checkpoints = list((run_directory / "checkpoints").glob("best_epoch_*.pt"))
    if not checkpoints:
        raise RuntimeError("protein workflow did not produce a best checkpoint")

    def epoch(path: Path) -> int:
        return int(path.stem.removeprefix("best_epoch_"))

    selected = max(checkpoints, key=epoch)
    return _capture_private_training_output(
        selected,
        run_directory=run_directory,
        label="trained energy checkpoint",
    )


def _checkpoint_identity(run_directory: Path) -> tuple[str, str]:
    snapshot = _checkpoint_snapshot(run_directory)
    return snapshot.path.name, snapshot.sha256


def _workflow_artifact_snapshots(
    run_directory: Path,
) -> tuple[dict[str, dict[str, str]], dict[str, RegularFileSnapshot]]:
    relative_paths = {
        "test_fasta": Path("data_splits/test.fasta"),
        "baseline_fasta": Path("baseline/proposal_only_generated_sequences.fasta"),
        "guided_fasta": Path("guided/energy_guided_generated_sequences.fasta"),
        "training_history": Path("history.json"),
        "sequence_metrics": Path("baseline_vs_guided.json"),
        "baseline_quality": Path("baseline_eval/quality_summary.json"),
        "guided_quality": Path("guided_eval/quality_summary.json"),
    }
    identities: dict[str, dict[str, str]] = {}
    snapshots: dict[str, RegularFileSnapshot] = {}
    for name, relative_path in relative_paths.items():
        path = run_directory / relative_path
        try:
            snapshot = _capture_private_training_output(
                path,
                run_directory=run_directory,
                label=f"protein workflow artifact {name}",
            )
        except ValueError:
            raise RuntimeError(
                f"protein workflow did not produce a stable {relative_path}"
            ) from None
        identities[name] = {
            "relative_path": relative_path.as_posix(),
            "sha256": snapshot.sha256,
        }
        snapshots[name] = snapshot
    return identities, snapshots


def _workflow_artifact_identities(run_directory: Path) -> dict[str, dict[str, str]]:
    identities, _ = _workflow_artifact_snapshots(run_directory)
    return identities


def _capture_training_outputs(
    run_directory: Path,
) -> tuple[str, str, dict[str, dict[str, str]]]:
    """Capture one internally consistent set of stable training outputs."""

    name, digest, artifacts, _ = _capture_training_output_snapshots(run_directory)
    return name, digest, artifacts


def _capture_training_output_snapshots(
    run_directory: Path,
) -> tuple[
    str,
    str,
    dict[str, dict[str, str]],
    dict[str, RegularFileSnapshot],
]:
    """Capture identities plus snapshots retained until record publication."""

    checkpoint = _checkpoint_snapshot(run_directory)
    artifacts, snapshots = _workflow_artifact_snapshots(run_directory)
    revalidate_regular_file(checkpoint, label="trained energy checkpoint")
    for name, snapshot in snapshots.items():
        revalidate_regular_file(snapshot, label=f"protein workflow artifact {name}")
    snapshots["trained_energy_checkpoint"] = checkpoint
    return checkpoint.path.name, checkpoint.sha256, artifacts, snapshots


def revalidate_training_output_snapshots(
    snapshots: dict[str, RegularFileSnapshot],
) -> None:
    """Require every captured training output to remain at record publication."""

    expected = {
        "trained_energy_checkpoint",
        "test_fasta",
        "baseline_fasta",
        "guided_fasta",
        "training_history",
        "sequence_metrics",
        "baseline_quality",
        "guided_quality",
    }
    if set(snapshots) != expected:
        raise ValueError("complete training output snapshots are required")
    for name, snapshot in snapshots.items():
        revalidate_regular_file(snapshot, label=f"training output {name}")


def _receipt_records(sampler: KaiwuSampler) -> list[dict[str, Any]]:
    return [
        {
            "schema": receipt.schema,
            "task_name": receipt.task_name,
            "matrix_sha256": receipt.matrix_sha256,
            "matrix_size": receipt.matrix_size,
            "mode": receipt.mode,
            "requested_samples": receipt.requested_samples,
            "project_no": receipt.project_no,
            "submitted_at": receipt.submitted_at,
            "provider_task_id": receipt.provider_task_id,
            "provider_target": receipt.provider_target,
        }
        for receipt in sampler.receipts
    ]


def _training_component_ready(record: dict[str, Any]) -> bool:
    """Return whether one seed is eligible for final-component validation."""

    remote_calls = record.get("remote_call_count")
    per_seed_budget = record.get("protein_remote_call_budget_per_seed")
    receipts = record.get("task_receipts")
    task_ids = record.get("qboson_task_ids")
    checkpoint_sha256 = record.get("trained_energy_checkpoint_sha256")
    return bool(
        record.get("run_completed") is True
        and record.get("failure") is None
        and record.get("artifact_inputs_unchanged") is True
        and record.get("transport") == "kaiwu_cim"
        and all(
            record.get(field) is True
            for field in (
                "pinned_sdk_client",
                "real_provider_evidence",
                "qboson_hardware_used",
                "provider_identity_complete",
                "provider_reported_target",
                "precision_evidence_complete",
                "secrets_redacted",
            )
        )
        and record.get("fallback_occurred") is False
        and isinstance(record.get("qboson_target"), str)
        and bool(record["qboson_target"].strip())
        and type(remote_calls) is int
        and remote_calls > 0
        and type(per_seed_budget) is int
        and per_seed_budget > 0
        and remote_calls <= per_seed_budget
        and isinstance(receipts, list)
        and len(receipts) == remote_calls
        and isinstance(task_ids, list)
        and len(task_ids) == remote_calls
        and len(task_ids) == len(set(task_ids))
        and isinstance(checkpoint_sha256, str)
        and re.fullmatch(r"[0-9a-f]{64}", checkpoint_sha256) is not None
    )


def run_training_seed(
    *,
    workflow: Any,
    config: dict[str, Any],
    config_sha256: str,
    sampler: KaiwuSampler,
    dataset_path: Path,
    checkpoint_path: Path,
    output_root: Path,
    seed: int,
    execution_host: str,
    observed_hostname: str,
    observed_gpu: str,
    source_revision: str,
    plugin_revision: str,
    source_preflight_sha256: str,
    transfer_manifest_sha256: str,
    environment_lock_sha256: str,
    provider_resource_gate: dict[str, Any],
    sdk_version: str,
    preflight_sha256: str,
    artifact_snapshots: dict[str, ArtifactSnapshot] | None = None,
    dataset_source_snapshot: RegularFileSnapshot | None = None,
    retained_output_snapshots: dict[str, RegularFileSnapshot] | None = None,
) -> dict[str, Any]:
    provider_resource_gate = validate_provider_resource_gate(provider_resource_gate)
    if (artifact_snapshots is None) != (dataset_source_snapshot is None):
        raise ValueError(
            "artifact and dataset source snapshots must be supplied together"
        )
    failure: dict[str, str] | None = None
    run_directory: Path | None = None
    checkpoint_name: str | None = None
    checkpoint_sha256: str | None = None
    workflow_artifacts: dict[str, dict[str, str]] = {}
    try:
        if artifact_snapshots is not None:
            revalidate_artifact_snapshots(artifact_snapshots)
        if dataset_source_snapshot is not None:
            revalidate_regular_file(
                dataset_source_snapshot, label="dataset source archive"
            )
        workflow_config = _build_workflow_config(
            workflow,
            config,
            dataset_path=dataset_path,
            checkpoint_path=checkpoint_path,
            seed=seed,
        )
        run_directory = _run_workflow(
            workflow, workflow_config, sampler, output_root / f"seed-{seed}"
        )
        if artifact_snapshots is not None:
            revalidate_artifact_snapshots(artifact_snapshots)
        if dataset_source_snapshot is not None:
            revalidate_regular_file(
                dataset_source_snapshot, label="dataset source archive"
            )
        (
            checkpoint_name,
            checkpoint_sha256,
            workflow_artifacts,
            output_snapshots,
        ) = _capture_training_output_snapshots(run_directory)
        if retained_output_snapshots is not None:
            retained_output_snapshots.clear()
            retained_output_snapshots.update(output_snapshots)
    except (Exception, KeyboardInterrupt) as exc:
        failure = redacted_failure_record(exc)

    receipts = _receipt_records(sampler)
    provider_identity_complete = bool(receipts) and all(
        bool(receipt["provider_task_id"]) and bool(receipt["provider_target"])
        for receipt in receipts
    )
    targets = {
        receipt["provider_target"]
        for receipt in receipts
        if isinstance(receipt["provider_target"], str)
    }
    requested_sample_counts = {
        receipt["requested_samples"]
        for receipt in receipts
        if type(receipt["requested_samples"]) is int
    }
    precision_reports = sampler.precision_reports
    precision_evidence = sampler.precision_evidence
    precision_evidence_complete = _precision_evidence_complete(sampler, receipts)
    verified_provider_transport = type(sampler.client) is KaiwuSDKClient
    qboson_hardware_used = bool(
        verified_provider_transport
        and sampler.remote_call_count > 0
        and _completed_provider_result_has_identity(sampler)
    )
    return {
        "schema": SCHEMA,
        "version": "1.0",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "experiment_config_sha256": config_sha256,
        "artifact_preflight_sha256": preflight_sha256,
        "source_revision": source_revision,
        "flagquantum_version": flagquantum_version,
        "kaiwu_pytorch_plugin_revision": plugin_revision,
        "source_preflight_sha256": source_preflight_sha256,
        "transfer_manifest_sha256": transfer_manifest_sha256,
        "environment_lock_sha256": environment_lock_sha256,
        "provider_resource_gate": provider_resource_gate,
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "kaiwu_sdk_version": sdk_version,
        "execution_host": execution_host,
        "observed_hostname": observed_hostname,
        "observed_gpu_model": observed_gpu,
        "requested_cuda_device": "cuda:0",
        "seed": seed,
        "transport": ("kaiwu_cim" if verified_provider_transport else "injected_test"),
        "pinned_sdk_client": verified_provider_transport,
        "real_provider_evidence": qboson_hardware_used,
        "qboson_hardware_used": qboson_hardware_used,
        "provider_reported_target": provider_identity_complete and len(targets) == 1,
        "qboson_target": next(iter(targets)) if len(targets) == 1 else None,
        "qboson_task_ids": [
            receipt["provider_task_id"]
            for receipt in receipts
            if isinstance(receipt["provider_task_id"], str)
        ],
        "sampling_mode": "sampling",
        "requested_samples": (
            next(iter(requested_sample_counts))
            if len(requested_sample_counts) == 1
            else None
        ),
        "fallback_occurred": False,
        "secrets_redacted": True,
        "run_completed": failure is None,
        "failure": failure,
        "remote_call_budget": config["remote_call_budget"],
        "protein_remote_call_budget_per_seed": config["training"][
            "remote_call_budget_per_seed"
        ],
        "estimated_worst_case_remote_calls": _estimate_protein_remote_calls(config),
        "remote_call_count": sampler.remote_call_count,
        "task_receipts": receipts,
        "provider_identity_complete": provider_identity_complete,
        "precision_report_count": len(precision_reports),
        "precision_policy": {
            "name": config["precision_policy"]["name"],
            "target_min": config["precision_policy"]["target_min"],
            "target_max": config["precision_policy"]["target_max"],
            "matrix_count": len(precision_reports),
            "scale_factor_min": min(
                (report.scale_factor for report in precision_reports), default=None
            ),
            "scale_factor_max": max(
                (report.scale_factor for report in precision_reports), default=None
            ),
            "max_abs_error": max(
                (report.max_abs_error for report in precision_reports), default=None
            ),
            "mean_of_matrix_mean_abs_error": (
                sum(report.mean_abs_error for report in precision_reports)
                / len(precision_reports)
                if precision_reports
                else None
            ),
        },
        "precision_evidence_complete": precision_evidence_complete,
        "precision_evidence": [asdict(evidence) for evidence in precision_evidence],
        "run_directory_name": run_directory.name if run_directory else None,
        "trained_energy_checkpoint_name": checkpoint_name,
        "trained_energy_checkpoint_sha256": checkpoint_sha256,
        "workflow_artifacts": workflow_artifacts,
        "acceptance": {
            "system": "not_evaluated",
            "application": "not_evaluated",
        },
        "limitations": [
            "This record covers one protein-training seed only.",
            "ESM2 evaluation and two-host system acceptance are separate gates.",
            "No performance, distributed, domestic-accelerator, or quantum-advantage claim is made.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--plugin-root", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--dataset-source-archive", required=True, type=Path)
    parser.add_argument("--base-checkpoint", required=True, type=Path)
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--evaluation-model", required=True, type=Path)
    parser.add_argument("--artifact-preflight-output", required=True, type=Path)
    parser.add_argument("--sdk-checkpoint-dir", required=True, type=Path)
    parser.add_argument("--workflow-output-root", required=True, type=Path)
    parser.add_argument("--run-record", required=True, type=Path)
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
    parser.add_argument("--provider-resources", required=True, type=Path)
    parser.add_argument("--project-no")
    parser.add_argument("--task-prefix", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--expected-sdk-version", choices=("1.3.1",), default="1.3.1")
    parser.add_argument("--requested-samples", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument("--poll-interval", type=float, default=60.0)
    parser.add_argument("--acknowledge-provider-cost", required=True)
    args = parser.parse_args()
    if args.acknowledge_provider_cost != ACKNOWLEDGEMENT:
        parser.error("invalid provider-cost acknowledgement; no task was submitted")
    try:
        if args.project_no is not None:
            args.project_no = normalize_provider_identifier(
                args.project_no, label="--project-no"
            )
        args.task_prefix = normalize_provider_identifier(
            args.task_prefix, label="--task-prefix"
        )
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))
    path_arguments = {
        "config": args.config,
        "plugin-root": args.plugin_root,
        "dataset": args.dataset,
        "dataset-source-archive": args.dataset_source_archive,
        "base-checkpoint": args.base_checkpoint,
        "tokenizer": args.tokenizer,
        "evaluation-model": args.evaluation_model,
        "artifact-preflight-output": args.artifact_preflight_output,
        "sdk-checkpoint-dir": args.sdk_checkpoint_dir,
        "workflow-output-root": args.workflow_output_root,
        "run-record": args.run_record,
        "source-preflight": args.source_preflight,
        "environment-lock": args.environment_lock,
        "provider-resources": args.provider_resources,
    }
    for label, path in path_arguments.items():
        if not path.is_absolute():
            parser.error(f"--{label} must be an absolute path")
    validate_private_json_output_path(args.artifact_preflight_output)
    validate_private_json_output_path(args.run_record)
    validate_private_directory(
        args.sdk_checkpoint_dir, label="Kaiwu checkpoint directory"
    )
    validate_private_directory(
        args.workflow_output_root, label="protein workflow output directory"
    )

    config, config_sha256 = _load_frozen_config(args.config)
    verify_approved_project_assignment(args.project_no, config["kaiwu_sdk"])
    if args.requested_samples != config["requested_samples"]:
        parser.error("--requested-samples differs from the frozen configuration")
    if args.seed not in config["seeds"]:
        parser.error("--seed is not present in the frozen seed list")
    provider_resources, provider_resources_sha256 = load_provider_resources(
        args.provider_resources
    )
    provider_resources_checked_at = datetime.now(timezone.utc)
    resources_ready, resource_reason = assess_provider_budget(
        provider_resources,
        mode="sampling",
        required_calls=config["training"]["remote_call_budget_per_seed"],
        now=provider_resources_checked_at,
    )
    if not resources_ready:
        parser.error(f"provider resource gate failed: {resource_reason}")
    provider_resource_gate = build_provider_resource_gate(
        snapshot_sha256=provider_resources_sha256,
        checked_at=provider_resources_checked_at,
        mode="sampling",
        required_calls=config["training"]["remote_call_budget_per_seed"],
    )
    if args.expected_hostname != config["host_identities"][args.execution_host]:
        parser.error("--expected-hostname differs from the frozen host identity")
    observed_hostname = socket.gethostname()
    if observed_hostname != args.expected_hostname:
        parser.error("observed hostname differs from --expected-hostname")
    device = torch.device("cuda:0")
    if not torch.cuda.is_available():
        parser.error("protein training requires CUDA")
    torch.cuda.set_device(device)
    observed_gpu = torch.cuda.get_device_name(device)
    if "A800" not in observed_gpu:
        parser.error("protein training requires an NVIDIA A800")
    role, _, target_range = _validate_lane(
        config,
        execution_host=args.execution_host,
        source_revision=args.source_revision,
        plugin_revision=args.plugin_revision,
        sdk_version=args.expected_sdk_version,
    )
    if role != "primary":
        parser.error("the full protein training workflow runs only on primary_host")
    if args.base_checkpoint.resolve() != args.tokenizer.resolve():
        parser.error("the pinned plugin requires tokenizer files in base checkpoint")
    source_preflight, source_preflight_sha256 = load_source_preflight(
        args.source_preflight,
        execution_host=args.execution_host,
        source_revision=args.source_revision,
        plugin_revision=args.plugin_revision,
        source_root=Path(__file__).resolve().parents[2],
        plugin_root=args.plugin_root,
    )
    environment_record, environment_lock_sha256 = verify_frozen_environment_lock(
        args.environment_lock,
        expected_sha256=config["software"]["environment_lock_sha256"],
    )
    verify_approved_kaiwu_distribution(environment_record, config["kaiwu_sdk"])

    artifact_paths = {
        "dataset": args.dataset,
        "base_checkpoint": args.base_checkpoint,
        "tokenizer": args.tokenizer,
        "evaluation_model": args.evaluation_model,
    }
    (
        artifact_preflight,
        artifact_snapshots,
        dataset_source_snapshot,
    ) = preflight_artifacts_with_snapshots(
        args.config,
        artifact_paths,
        args.artifact_preflight_output,
        dataset_source_archive=args.dataset_source_archive,
    )
    if artifact_preflight.get("config_sha256") != config_sha256:
        raise RuntimeError("frozen experiment config changed before training")
    preflight_sha256 = hashlib.sha256(
        read_private_bytes(
            args.artifact_preflight_output,
            label="artifact preflight",
            max_bytes=_MAX_ARTIFACT_PREFLIGHT_BYTES,
        )
    ).hexdigest()

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    workflow = _load_pinned_workflow(args.plugin_root)
    provider_resources_checked_at = datetime.now(timezone.utc)
    resources_ready, resource_reason = assess_provider_budget(
        provider_resources,
        mode="sampling",
        required_calls=config["training"]["remote_call_budget_per_seed"],
        now=provider_resources_checked_at,
    )
    if not resources_ready:
        parser.error(f"provider resource gate failed: {resource_reason}")
    provider_resource_gate = build_provider_resource_gate(
        snapshot_sha256=provider_resources_sha256,
        mode="sampling",
        required_calls=config["training"]["remote_call_budget_per_seed"],
        checked_at=provider_resources_checked_at,
    )
    user_id, sdk_code = resolve_kaiwu_credentials()
    credentials = KaiwuCredentials(user_id=user_id, sdk_code=sdk_code)
    os.environ.pop("QBOSON_USER_ID", None)
    os.environ.pop("QBOSON_SDK_CODE", None)
    client = KaiwuSDKClient(
        checkpoint_dir=args.sdk_checkpoint_dir,
        credentials=credentials,
        expected_version=args.expected_sdk_version,
        submission_deadline=provider_resource_valid_until(provider_resources),
    )
    sampler = KaiwuSampler(
        client=client,
        task_name=f"{args.task_prefix}-seed-{args.seed}",
        project_no=args.project_no,
        requested_samples=args.requested_samples,
        timeout=args.timeout,
        poll_interval=args.poll_interval,
        max_remote_calls=config["training"]["remote_call_budget_per_seed"],
        integer_target_range=target_range,
    )
    output_snapshots: dict[str, RegularFileSnapshot] = {}
    payload = run_training_seed(
        workflow=workflow,
        config=config,
        config_sha256=config_sha256,
        sampler=sampler,
        dataset_path=args.dataset,
        checkpoint_path=args.base_checkpoint,
        output_root=args.workflow_output_root,
        seed=args.seed,
        execution_host=args.execution_host,
        observed_hostname=observed_hostname,
        observed_gpu=observed_gpu,
        source_revision=args.source_revision,
        plugin_revision=args.plugin_revision,
        source_preflight_sha256=source_preflight_sha256,
        transfer_manifest_sha256=source_preflight["manifest_sha256"],
        environment_lock_sha256=environment_lock_sha256,
        provider_resource_gate=provider_resource_gate,
        sdk_version=args.expected_sdk_version,
        preflight_sha256=preflight_sha256,
        artifact_snapshots=artifact_snapshots,
        dataset_source_snapshot=dataset_source_snapshot,
        retained_output_snapshots=output_snapshots,
    )
    artifact_postflight_error: BaseException | None = None
    try:
        revalidate_artifact_snapshots(artifact_snapshots)
        revalidate_regular_file(dataset_source_snapshot, label="dataset source archive")
        assert_artifacts_unchanged(
            args.config,
            artifact_paths,
            artifact_preflight,
            dataset_source_archive=args.dataset_source_archive,
        )
        revalidate_regular_file(dataset_source_snapshot, label="dataset source archive")
    except (OSError, ValueError) as exc:
        artifact_postflight_error = exc
    apply_artifact_postflight(payload, artifact_postflight_error)
    try:
        if output_snapshots:
            revalidate_training_output_snapshots(output_snapshots)
        elif payload.get("run_completed") is True:
            raise ValueError("completed training run has no retained output snapshots")
    except (OSError, ValueError) as exc:
        payload["run_completed"] = False
        if payload.get("failure") is None:
            payload["failure"] = redacted_failure_record(exc)
    _write_private_redacted_json(
        args.run_record,
        payload,
        forbidden_values=(user_id, sdk_code),
    )
    print(f"Private protein-training record written to {args.run_record}")
    if not _training_component_ready(payload):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
