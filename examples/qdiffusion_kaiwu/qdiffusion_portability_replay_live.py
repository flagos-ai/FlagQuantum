"""Replay one frozen trained QDiffusion fixture on the second A800 host.

This command consumes QBoson quota. It is a bounded portability gate, not a
second training run and not standalone application acceptance.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import math
import os
import platform
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
from examples.qdiffusion_kaiwu.preflight_protein_artifacts import (
    AMINO_ACIDS,
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
    validate_provider_resource_gate,
)
from examples.qdiffusion_kaiwu.qdiffusion_protein_evaluate import (
    _load_training_record,
    _revalidate_training_paths,
    _verified_training_paths,
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
    _receipt_records,
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
    revalidate_regular_file,
)
from examples.qdiffusion_kaiwu.verify_environment_lock import (
    verify_frozen_environment_lock,
)
from flagquantum.ecosystem.kaiwu import KaiwuSampler
from flagquantum.remote.kaiwu import (
    KaiwuCredentials,
    KaiwuSDKClient,
    KaiwuTaskClient,
    resolve_kaiwu_credentials,
)
from flagquantum.version import __version__ as flagquantum_version

SCHEMA = "flagquantum.qboson_qdiffusion_portability_replay"
_MAX_ARTIFACT_PREFLIGHT_BYTES = 4 * 1024 * 1024


def _load_pinned_modules(
    plugin_root: Path,
) -> tuple[ModuleType, ModuleType, ModuleType]:
    root = plugin_root.resolve()
    case_root = root / "example" / "qdiffusion"
    source_root = root / "src"
    expected = {
        "builder": case_root / "dplm" / "utils" / "dplm_builder.py",
        "runtime": case_root / "dplm" / "utils" / "runtime.py",
        "io": case_root / "dplm" / "utils" / "io.py",
    }
    if not source_root.is_dir() or any(
        not path.is_file() for path in expected.values()
    ):
        raise ValueError("plugin root does not contain the pinned DPLM modules")
    _load_pinned_qdiffusion_api(root)
    for path in (source_root, case_root):
        encoded = str(path)
        if encoded not in sys.path:
            sys.path.insert(0, encoded)
    modules = {
        "builder": importlib.import_module("dplm.utils.dplm_builder"),
        "runtime": importlib.import_module("dplm.utils.runtime"),
        "io": importlib.import_module("dplm.utils.io"),
    }
    for name, module in modules.items():
        if Path(str(module.__file__)).resolve() != expected[name].resolve():
            raise RuntimeError(f"imported DPLM {name} module is outside --plugin-root")
    _validate_imported_module_tree(
        module_prefix="dplm",
        expected_root=case_root / "dplm",
        label="DPLM",
    )
    return modules["builder"], modules["runtime"], modules["io"]


def _verified_checkpoint_snapshot(
    run_directory: Path,
    checkpoint_path: Path,
    training_record: dict[str, Any],
) -> RegularFileSnapshot:
    validate_private_directory(
        run_directory, label="protein training run directory"
    )
    try:
        checkpoint_path.resolve().relative_to(run_directory.resolve())
    except ValueError:
        raise ValueError(
            "trained checkpoint is outside the recorded run directory"
        ) from None
    if checkpoint_path.name != training_record.get("trained_energy_checkpoint_name"):
        raise ValueError("trained checkpoint name differs from the training record")
    try:
        snapshot = capture_regular_file(
            checkpoint_path, label="trained energy checkpoint"
        )
    except ValueError:
        raise ValueError("trained checkpoint is absent or symbolic") from None
    if snapshot.sha256 != training_record.get("trained_energy_checkpoint_sha256"):
        raise ValueError("trained checkpoint digest differs from the training record")
    if stat.S_IMODE(checkpoint_path.lstat().st_mode) & 0o077:
        raise ValueError("trained checkpoint must be owner-only")
    revalidate_regular_file(snapshot, label="trained energy checkpoint")
    return snapshot


def _verified_checkpoint(
    run_directory: Path,
    checkpoint_path: Path,
    training_record: dict[str, Any],
) -> str:
    return _verified_checkpoint_snapshot(
        run_directory, checkpoint_path, training_record
    ).sha256


def _token_constraints(
    generator: Any, generated: torch.Tensor
) -> tuple[bool, int, str]:
    decoded = generator.tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
    sequence = decoded.replace(" ", "").strip().upper()
    valid = bool(sequence) and not bool(set(sequence) - AMINO_ACIDS)
    return valid, len(sequence), hashlib.sha256(sequence.encode()).hexdigest()


def run_portability_replay(
    *,
    builder: Any,
    runtime: Any,
    io_module: Any,
    client: KaiwuTaskClient,
    config: dict[str, Any],
    config_sha256: str,
    artifact_preflight_sha256: str,
    training_record_sha256: str,
    test_fasta: Path,
    base_checkpoint: Path,
    trained_checkpoint: Path,
    trained_checkpoint_sha256: str,
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
    project_no: str,
    task_prefix: str,
    requested_samples: int,
    timeout: float,
    poll_interval: float,
    device: torch.device,
    real_provider_transport: bool,
    trained_checkpoint_snapshot: RegularFileSnapshot | None = None,
    artifact_snapshots: dict[str, ArtifactSnapshot] | None = None,
) -> dict[str, Any]:
    provider_resource_gate = validate_provider_resource_gate(provider_resource_gate)
    if requested_samples != config.get("requested_samples"):
        raise ValueError("requested_samples differs from the frozen configuration")
    generation = config["generation"]
    target_range = (
        config["precision_policy"]["target_min"],
        config["precision_policy"]["target_max"],
    )
    sampler = KaiwuSampler(
        client=client,
        task_name=f"{task_prefix}-portability",
        project_no=project_no,
        requested_samples=requested_samples,
        timeout=timeout,
        poll_interval=poll_interval,
        max_remote_calls=config["remote_call_budget"],
        integer_target_range=target_range,
    )
    failure: dict[str, str] | None = None
    energy_objective: float | None = None
    generated_device: str | None = None
    token_constraints_passed = False
    generated_length: int | None = None
    generated_sha256: str | None = None
    retrieval_resubmitted: bool | None = None
    checkpoint_snapshot = trained_checkpoint_snapshot or capture_regular_file(
        trained_checkpoint, label="trained energy checkpoint"
    )
    if (
        checkpoint_snapshot.path != trained_checkpoint
        or checkpoint_snapshot.sha256 != trained_checkpoint_sha256
    ):
        raise ValueError("trained checkpoint snapshot differs from replay inputs")
    try:
        if artifact_snapshots is not None:
            revalidate_artifact_snapshots(artifact_snapshots)
        generator = (
            builder.build_qdiffusion(
                proposal_ckpt=str(base_checkpoint),
                energy_ckpt=str(base_checkpoint),
                bm_sampler=sampler,
                bm_sampler_type="flagquantum-kaiwu",
                bm_sampler_kwargs={},
                num_candidates=generation["num_candidates"],
                proposal_temperature=generation["proposal_temperature"],
                proposal_noise_scale=generation["proposal_noise_scale"],
                energy_temperature=generation["energy_temperature"],
                disable_resample=generation["disable_resample"],
                resample_ratio=generation["resample_ratio"],
                resample_top_p=generation["resample_top_p"],
                freeze_proposal=config["training"]["freeze_proposal"],
            )
            .eval()
            .to(device)
        )
        if artifact_snapshots is not None:
            revalidate_artifact_snapshots(artifact_snapshots)
        if id(getattr(generator.energy_model, "sampler", None)) != id(sampler):
            raise RuntimeError("DPLM builder did not retain the FlagQuantum sampler")
        revalidate_regular_file(
            checkpoint_snapshot, label="trained energy checkpoint"
        )
        runtime.load_trained_energy_weights(
            generator, str(trained_checkpoint), str(device)
        )
        revalidate_regular_file(
            checkpoint_snapshot, label="trained energy checkpoint"
        )
        records = io_module.read_fasta_records(test_fasta)
        index = generation["portability_fixture_index"]
        if len(records) != generation["sequence_count"]:
            raise RuntimeError("recorded test FASTA count differs from frozen config")
        _, sequence = records[index]
        runtime.seed_torch(generation["portability_training_seed"])
        target = runtime.encode_sequence(
            generator, sequence, max_length=config["dataset"]["max_length"]
        )
        with torch.no_grad():
            objective = generator.objective({"targets": target})
            generated = generator.generate(
                target,
                max_steps=generation["portability_steps"],
            )
        energy_objective = float(objective["energy_objective"].mean().item())
        if not math.isfinite(energy_objective):
            raise RuntimeError("portability energy objective is not finite")
        generated_device = str(generated.device)
        (
            token_constraints_passed,
            generated_length,
            generated_sha256,
        ) = _token_constraints(generator, generated)
        revalidate_regular_file(
            checkpoint_snapshot, label="trained energy checkpoint"
        )
        if artifact_snapshots is not None:
            revalidate_artifact_snapshots(artifact_snapshots)
        last_job = sampler.last_job
        if last_job is None:
            raise RuntimeError("portability replay completed without a recoverable job")
        call_count = sampler.remote_call_count
        receipt = last_job.receipt
        repeated = last_job.result()
        retrieval_resubmitted = not (
            sampler.remote_call_count == call_count and repeated.receipt == receipt
        )
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
    precision_reports = sampler.precision_reports
    precision_evidence = sampler.precision_evidence
    precision_complete = _precision_evidence_complete(sampler, receipts)
    verified_transport = real_provider_transport and type(client) is KaiwuSDKClient
    provider_use_proven = bool(
        verified_transport
        and sampler.remote_call_count > 0
        and _completed_provider_result_has_identity(sampler)
    )
    run_completed = failure is None
    portability_pass = bool(
        run_completed
        and verified_transport
        and provider_use_proven
        and provider_identity_complete
        and len(targets) == 1
        and retrieval_resubmitted is False
        and device == torch.device("cuda:0")
        and "A800" in observed_gpu
        and token_constraints_passed
        and generated_device == "cuda:0"
        and isinstance(energy_objective, float)
        and math.isfinite(energy_objective)
        and precision_complete
    )
    return {
        "schema": SCHEMA,
        "version": "1.0",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "experiment_config_sha256": config_sha256,
        "artifact_preflight_sha256": artifact_preflight_sha256,
        "training_record_sha256": training_record_sha256,
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
        "requested_cuda_device": str(device),
        "observed_tensor_device": generated_device,
        "transport": "kaiwu_cim" if verified_transport else "injected_test",
        "pinned_sdk_client": verified_transport,
        "qboson_hardware_used": provider_use_proven,
        "real_provider_evidence": provider_use_proven,
        "provider_identity_complete": provider_identity_complete,
        "provider_reported_target": provider_identity_complete and len(targets) == 1,
        "qboson_target": next(iter(targets)) if len(targets) == 1 else None,
        "qboson_task_ids": [
            receipt["provider_task_id"]
            for receipt in receipts
            if isinstance(receipt["provider_task_id"], str)
        ],
        "task_receipts": receipts,
        "sampling_mode": "sampling",
        "requested_samples": requested_samples,
        "returned_samples": (
            len(sampler.last_result.samples)
            if sampler.last_result is not None
            else None
        ),
        "remote_call_budget": config["remote_call_budget"],
        "remote_call_count": sampler.remote_call_count,
        "fallback_occurred": False,
        "retrieval_resubmitted": retrieval_resubmitted,
        "secrets_redacted": True,
        "precision_policy": {
            "name": config["precision_policy"]["name"],
            "target_min": target_range[0],
            "target_max": target_range[1],
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
        "precision_evidence": [asdict(evidence) for evidence in precision_evidence],
        "precision_evidence_complete": precision_complete,
        "trained_energy_checkpoint_sha256": trained_checkpoint_sha256,
        "artifacts": {
            "dataset_sha256": config["dataset"]["sha256"],
            "base_checkpoint_sha256": config["checkpoint"]["sha256"],
            "tokenizer_sha256": config["tokenizer"]["sha256"],
            "evaluation_model_sha256": config["evaluation_model"]["sha256"],
            "trained_energy_checkpoint_sha256": trained_checkpoint_sha256,
        },
        "fixture": {
            "training_seed": generation["portability_training_seed"],
            "index": generation["portability_fixture_index"],
            "steps": generation["portability_steps"],
            "energy_objective": energy_objective,
            "generated_length": generated_length,
            "generated_sha256": generated_sha256,
            "token_constraints_passed": token_constraints_passed,
        },
        "run_completed": run_completed,
        "failure": failure,
        "acceptance": {"portability": "pass" if portability_pass else "fail"},
        "limitations": [
            "This is one fixed replay fixture, not a second training run.",
            "Final acceptance also requires both system gates and all primary-host seeds.",
            "No performance, distributed, domestic-accelerator, or quantum-advantage claim is made.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--plugin-root", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--base-checkpoint", required=True, type=Path)
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--evaluation-model", required=True, type=Path)
    parser.add_argument("--artifact-preflight-output", required=True, type=Path)
    parser.add_argument("--training-record", required=True, type=Path)
    parser.add_argument("--training-run-directory", required=True, type=Path)
    parser.add_argument("--trained-checkpoint", required=True, type=Path)
    parser.add_argument("--sdk-checkpoint-dir", required=True, type=Path)
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
    parser.add_argument("--provider-resources", required=True, type=Path)
    parser.add_argument("--project-no", required=True)
    parser.add_argument("--task-prefix", required=True)
    parser.add_argument("--expected-sdk-version", choices=("1.3.1",), default="1.3.1")
    parser.add_argument("--requested-samples", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument("--poll-interval", type=float, default=60.0)
    parser.add_argument("--acknowledge-provider-cost", required=True)
    args = parser.parse_args()
    if args.acknowledge_provider_cost != ACKNOWLEDGEMENT:
        parser.error("invalid provider-cost acknowledgement; no task was submitted")
    try:
        args.project_no = normalize_provider_identifier(
            args.project_no, label="--project-no"
        )
        args.task_prefix = normalize_provider_identifier(
            args.task_prefix, label="--task-prefix"
        )
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))
    for label, path in {
        "config": args.config,
        "plugin-root": args.plugin_root,
        "dataset": args.dataset,
        "base-checkpoint": args.base_checkpoint,
        "tokenizer": args.tokenizer,
        "evaluation-model": args.evaluation_model,
        "artifact-preflight-output": args.artifact_preflight_output,
        "training-record": args.training_record,
        "training-run-directory": args.training_run_directory,
        "trained-checkpoint": args.trained_checkpoint,
        "sdk-checkpoint-dir": args.sdk_checkpoint_dir,
        "output": args.output,
        "source-preflight": args.source_preflight,
        "environment-lock": args.environment_lock,
        "provider-resources": args.provider_resources,
    }.items():
        if not path.is_absolute():
            parser.error(f"--{label} must be an absolute path")
    validate_private_json_output_path(args.artifact_preflight_output)
    validate_private_json_output_path(args.output)
    validate_private_directory(
        args.sdk_checkpoint_dir, label="Kaiwu checkpoint directory"
    )

    config, config_sha256 = _load_frozen_config(args.config)
    if args.requested_samples != config["requested_samples"]:
        parser.error("--requested-samples differs from the frozen configuration")
    provider_resources, provider_resources_sha256 = load_provider_resources(
        args.provider_resources
    )
    provider_resources_checked_at = datetime.now(timezone.utc)
    resources_ready, resource_reason = assess_provider_budget(
        provider_resources,
        mode="sampling",
        required_calls=config["remote_call_budget"],
        now=provider_resources_checked_at,
    )
    if not resources_ready:
        parser.error(f"provider resource gate failed: {resource_reason}")
    provider_resource_gate = build_provider_resource_gate(
        snapshot_sha256=provider_resources_sha256,
        checked_at=provider_resources_checked_at,
        mode="sampling",
        required_calls=config["remote_call_budget"],
    )
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
    if role != "portability_replay":
        parser.error("protein portability replay must run on replay_host")
    training_record, training_record_sha256 = _load_training_record(
        args.training_record
    )
    if training_record.get("experiment_config_sha256") != config_sha256:
        parser.error("training record belongs to another frozen configuration")
    if training_record.get("seed") != config["generation"]["portability_training_seed"]:
        parser.error("training record is not the frozen portability seed")
    if training_record.get("execution_host") != config["primary_host"]:
        parser.error("training record was not produced by primary_host")
    training_paths = _verified_training_paths(
        args.training_run_directory, training_record
    )
    checkpoint_snapshot = _verified_checkpoint_snapshot(
        args.training_run_directory, args.trained_checkpoint, training_record
    )
    checkpoint_sha256 = checkpoint_snapshot.sha256
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
    artifact_preflight, artifact_snapshots = preflight_artifacts_with_snapshots(
        args.config, artifact_paths, args.artifact_preflight_output
    )
    if artifact_preflight.get("config_sha256") != config_sha256:
        raise RuntimeError("frozen experiment config changed before portability replay")
    artifact_preflight_sha256 = hashlib.sha256(
        read_private_bytes(
            args.artifact_preflight_output,
            label="artifact preflight",
            max_bytes=_MAX_ARTIFACT_PREFLIGHT_BYTES,
        )
    ).hexdigest()

    device = torch.device("cuda:0")
    if not torch.cuda.is_available():
        parser.error("protein portability replay requires CUDA")
    torch.cuda.set_device(device)
    gpu = torch.cuda.get_device_name(device)
    if "A800" not in gpu:
        parser.error("protein portability replay requires an NVIDIA A800")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    builder, runtime, io_module = _load_pinned_modules(args.plugin_root)
    user_id, sdk_code = resolve_kaiwu_credentials()
    credentials = KaiwuCredentials(user_id=user_id, sdk_code=sdk_code)
    os.environ.pop("QBOSON_USER_ID", None)
    os.environ.pop("QBOSON_SDK_CODE", None)
    client = KaiwuSDKClient(
        checkpoint_dir=args.sdk_checkpoint_dir,
        credentials=credentials,
        expected_version=args.expected_sdk_version,
    )
    payload = run_portability_replay(
        builder=builder,
        runtime=runtime,
        io_module=io_module,
        client=client,
        config=config,
        config_sha256=config_sha256,
        artifact_preflight_sha256=artifact_preflight_sha256,
        training_record_sha256=training_record_sha256,
        test_fasta=training_paths["test_fasta"],
        base_checkpoint=args.base_checkpoint,
        trained_checkpoint=args.trained_checkpoint,
        trained_checkpoint_sha256=checkpoint_sha256,
        execution_host=args.execution_host,
        observed_hostname=hostname,
        observed_gpu=gpu,
        source_revision=args.source_revision,
        plugin_revision=args.plugin_revision,
        source_preflight_sha256=source_preflight_sha256,
        transfer_manifest_sha256=source_preflight["manifest_sha256"],
        environment_lock_sha256=environment_lock_sha256,
        provider_resource_gate=provider_resource_gate,
        sdk_version=args.expected_sdk_version,
        project_no=args.project_no,
        task_prefix=args.task_prefix,
        requested_samples=args.requested_samples,
        timeout=args.timeout,
        poll_interval=args.poll_interval,
        device=device,
        real_provider_transport=True,
        trained_checkpoint_snapshot=checkpoint_snapshot,
        artifact_snapshots=artifact_snapshots,
    )
    artifact_postflight_error: BaseException | None = None
    try:
        revalidate_artifact_snapshots(artifact_snapshots)
        assert_artifacts_unchanged(args.config, artifact_paths, artifact_preflight)
        _revalidate_training_paths(training_paths)
        revalidate_regular_file(
            checkpoint_snapshot, label="trained energy checkpoint"
        )
        revalidate_artifact_snapshots(artifact_snapshots)
    except (OSError, ValueError) as exc:
        artifact_postflight_error = exc
        payload["acceptance"]["portability"] = "fail"
    apply_artifact_postflight(payload, artifact_postflight_error)
    _write_private_redacted_json(
        args.output,
        payload,
        forbidden_values=(user_id, sdk_code),
    )
    print(f"Private portability-replay record written to {args.output}")
    if payload["acceptance"]["portability"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
