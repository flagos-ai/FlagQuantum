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
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any

import torch

from examples.qdiffusion_kaiwu.preflight_protein_artifacts import (
    AMINO_ACIDS,
    preflight_artifacts,
)
from examples.qdiffusion_kaiwu.qdiffusion_protein_evaluate import (
    _load_training_record,
    _verified_training_paths,
)
from examples.qdiffusion_kaiwu.qdiffusion_system_development_probe import (
    _load_pinned_qdiffusion_api,
)
from examples.qdiffusion_kaiwu.qdiffusion_system_live import (
    ACKNOWLEDGEMENT,
    _load_frozen_config,
    _receipt_records,
    _validate_lane,
    _write_private_redacted_json,
)
from examples.qdiffusion_kaiwu.source_preflight import load_source_preflight
from flagquantum.ecosystem.kaiwu import KaiwuSampler
from flagquantum.remote.kaiwu import (
    KaiwuCredentials,
    KaiwuSDKClient,
    KaiwuTaskClient,
    resolve_kaiwu_credentials,
)

SCHEMA = "flagquantum.qboson_qdiffusion_portability_replay"


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
    return modules["builder"], modules["runtime"], modules["io"]


def _verified_checkpoint(
    run_directory: Path,
    checkpoint_path: Path,
    training_record: dict[str, Any],
) -> str:
    try:
        checkpoint_path.resolve().relative_to(run_directory.resolve())
    except ValueError:
        raise ValueError(
            "trained checkpoint is outside the recorded run directory"
        ) from None
    if checkpoint_path.is_symlink() or not checkpoint_path.is_file():
        raise ValueError("trained checkpoint is absent or symbolic")
    if checkpoint_path.name != training_record.get("trained_energy_checkpoint_name"):
        raise ValueError("trained checkpoint name differs from the training record")
    digest = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    if digest != training_record.get("trained_energy_checkpoint_sha256"):
        raise ValueError("trained checkpoint digest differs from the training record")
    return digest


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
    sdk_version: str,
    project_no: str,
    task_prefix: str,
    requested_samples: int,
    timeout: float,
    poll_interval: float,
    device: torch.device,
    real_provider_transport: bool,
) -> dict[str, Any]:
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
    try:
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
        if id(getattr(generator.energy_model, "sampler", None)) != id(sampler):
            raise RuntimeError("DPLM builder did not retain the FlagQuantum sampler")
        runtime.load_trained_energy_weights(
            generator, str(trained_checkpoint), str(device)
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
        failure = {"type": type(exc).__name__, "message": str(exc)}

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
    verified_transport = real_provider_transport and isinstance(client, KaiwuSDKClient)
    run_completed = failure is None
    portability_pass = bool(
        run_completed
        and verified_transport
        and provider_identity_complete
        and retrieval_resubmitted is False
        and token_constraints_passed
        and generated_device == "cuda:0"
        and precision_reports
        and len(precision_reports) >= len(receipts)
    )
    return {
        "schema": SCHEMA,
        "version": "1.0",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "experiment_config_sha256": config_sha256,
        "artifact_preflight_sha256": artifact_preflight_sha256,
        "training_record_sha256": training_record_sha256,
        "source_revision": source_revision,
        "kaiwu_pytorch_plugin_revision": plugin_revision,
        "source_preflight_sha256": source_preflight_sha256,
        "transfer_manifest_sha256": transfer_manifest_sha256,
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "kaiwu_sdk_version": sdk_version,
        "execution_host": execution_host,
        "observed_hostname": observed_hostname,
        "observed_gpu_model": observed_gpu,
        "requested_cuda_device": str(device),
        "observed_tensor_device": generated_device,
        "transport": "kaiwu_cim" if verified_transport else "injected_test",
        "qboson_hardware_used": verified_transport and run_completed,
        "real_provider_evidence": verified_transport and run_completed,
        "provider_identity_complete": provider_identity_complete,
        "provider_reported_target": provider_identity_complete,
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
    parser.add_argument("--project-no", required=True)
    parser.add_argument("--task-prefix", required=True)
    parser.add_argument("--expected-sdk-version", default="1.3.1")
    parser.add_argument("--requested-samples", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument("--poll-interval", type=float, default=60.0)
    parser.add_argument("--acknowledge-provider-cost", required=True)
    args = parser.parse_args()
    if args.acknowledge_provider_cost != ACKNOWLEDGEMENT:
        parser.error("invalid provider-cost acknowledgement; no task was submitted")
    if not args.project_no.strip() or not args.task_prefix.strip():
        parser.error("--project-no and --task-prefix must be non-empty")
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
    }.items():
        if not path.is_absolute():
            parser.error(f"--{label} must be an absolute path")

    config, config_sha256 = _load_frozen_config(args.config)
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
    checkpoint_sha256 = _verified_checkpoint(
        args.training_run_directory, args.trained_checkpoint, training_record
    )
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
    preflight_artifacts(
        args.config,
        {
            "dataset": args.dataset,
            "base_checkpoint": args.base_checkpoint,
            "tokenizer": args.tokenizer,
            "evaluation_model": args.evaluation_model,
        },
        args.artifact_preflight_output,
    )
    artifact_preflight_sha256 = hashlib.sha256(
        args.artifact_preflight_output.read_bytes()
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
        sdk_version=args.expected_sdk_version,
        project_no=args.project_no,
        task_prefix=args.task_prefix,
        requested_samples=args.requested_samples,
        timeout=args.timeout,
        poll_interval=args.poll_interval,
        device=device,
        real_provider_transport=True,
    )
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
