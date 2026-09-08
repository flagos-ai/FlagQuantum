#!/usr/bin/env python3
"""Measure synchronous versus pipelined sharded statevector exchange.

Launch with ``torchrun``. Gloo runs validate ordering and artifact contracts;
only NCCL/CUDA runs can set ``accelerator_overlap_measured=true``. This script
always emits development evidence and cannot promote an ISSUE-044 release.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import flagquantum as fq  # noqa: E402, I001
from flagquantum.runtime.executors.statevector.forward_executor import (  # noqa: E402
    execute_torch_distributed_statevector,
)

SCHEMA = "flagquantum.statevector.exchange_overlap.v1"


def build_workload(n_wires: int, layers: int, device: torch.device) -> fq.Circuit:
    if n_wires < 4 or layers < 1:
        raise ValueError("n_wires >= 4 and layers >= 1 required")
    last = n_wires - 1
    circuit = fq.Circuit(n_wires, device=device)
    for layer in range(layers):
        scale = float(layer + 1)
        circuit.rx(last, 0.07 * scale).rz(last, -0.03 * scale).ry(last, 0.05 * scale)
        circuit.h(last - 1).cx(last - 1, last)
        circuit.h(0).x(last).phase(last, 0.02 * scale).ry(last, -0.04 * scale)
    return circuit


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _run_once(
    circuit: fq.Circuit,
    *,
    device: torch.device,
    exchange_buffer_bytes: int,
    pipeline: bool,
) -> tuple[Any, float, float | None]:
    dist.barrier()
    _sync(device)
    start_event = end_event = None
    if device.type == "cuda":
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
    started = time.perf_counter()
    result = execute_torch_distributed_statevector(
        circuit,
        device=device,
        exchange_buffer_bytes=exchange_buffer_bytes,
        pipeline_pair_exchange=pipeline,
    )
    if end_event is not None:
        end_event.record()
    _sync(device)
    host_seconds = time.perf_counter() - started
    device_seconds = (
        float(start_event.elapsed_time(end_event)) / 1000.0
        if start_event is not None and end_event is not None
        else None
    )
    dist.barrier()
    return result, host_seconds, device_seconds


def _profile_once(
    circuit: fq.Circuit,
    *,
    device: torch.device,
    exchange_buffer_bytes: int,
    pipeline: bool,
    trace_path: Path | None,
) -> list[dict[str, Any]]:
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
    ) as profile:
        _run_once(
            circuit,
            device=device,
            exchange_buffer_bytes=exchange_buffer_bytes,
            pipeline=pipeline,
        )
    if trace_path is not None:
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        profile.export_chrome_trace(str(trace_path))
    events = sorted(
        profile.key_averages(),
        key=lambda event: float(event.self_device_time_total),
        reverse=True,
    )
    return [
        {
            "operator": event.key,
            "count": int(event.count),
            "self_device_time_us": float(event.self_device_time_total),
            "device_time_us": float(event.device_time_total),
            "self_cpu_time_us": float(event.self_cpu_time_total),
            "cpu_time_us": float(event.cpu_time_total),
        }
        for event in events[:30]
    ]


def _aggregate_rank_samples(rows: list[dict[str, Any]], key: str) -> list[float]:
    count = len(rows[0][key])
    return [max(float(row[key][index]) for row in rows) for index in range(count)]


def _timing(samples: list[float]) -> dict[str, Any]:
    mean = statistics.fmean(samples)
    return {
        "samples_seconds": samples,
        "sample_count": len(samples),
        "median_seconds": statistics.median(samples),
        "mean_seconds": mean,
        "coefficient_of_variation": (
            statistics.pstdev(samples) / mean if len(samples) > 1 and mean else 0.0
        ),
    }


def _paired_improvement(
    pipeline_samples: list[float], synchronous_samples: list[float]
) -> dict[str, Any]:
    improvements = [
        synchronous - pipeline
        for pipeline, synchronous in zip(pipeline_samples, synchronous_samples)
    ]
    mean = statistics.fmean(improvements)
    standard_error = (
        statistics.stdev(improvements) / len(improvements) ** 0.5
        if len(improvements) > 1
        else 0.0
    )
    lower = mean - 1.96 * standard_error
    upper = mean + 1.96 * standard_error
    return {
        "paired_improvements_seconds": improvements,
        "mean_improvement_seconds": mean,
        "confidence_level": 0.95,
        "confidence_interval_seconds": [lower, upper],
        "benefit_demonstrated": lower > 0.0,
    }


def validate_overlap_payload(payload: dict[str, Any]) -> tuple[str, ...]:
    blockers = []
    if payload.get("schema_version") != SCHEMA:
        blockers.append("invalid_schema")
    if payload.get("distribution_semantics") != "sharded_across_ranks":
        blockers.append("not_sharded_across_ranks")
    if payload.get("scalability_claim_allowed") is not False:
        blockers.append("development_artifact_must_reject_scalability_claim")
    if payload.get("release_gate_allowed") is not False:
        blockers.append("development_artifact_must_reject_release")
    if not payload.get("correctness", {}).get("passed"):
        blockers.append("pipeline_sync_mismatch")
    pipeline = payload.get("pipeline", {})
    synchronous = payload.get("synchronous", {})
    if int(pipeline.get("sample_count", 0)) < 2:
        blockers.append("insufficient_pipeline_samples")
    if int(synchronous.get("sample_count", 0)) < 2:
        blockers.append("insufficient_synchronous_samples")
    if int(payload.get("runtime", {}).get("pipeline_prefetch_count", 0)) <= 0:
        blockers.append("pipeline_did_not_prefetch")
    if payload.get("accelerator_overlap_measured"):
        if payload.get("backend") != "nccl":
            blockers.append("accelerator_overlap_requires_nccl")
        if "device_timing" not in payload:
            blockers.append("accelerator_overlap_requires_cuda_events")
    return tuple(blockers)


def run(args: argparse.Namespace) -> dict[str, Any] | None:
    if args.repetitions < 2 or args.warmup < 0:
        raise ValueError("repetitions >= 2 and warmup >= 0 required")
    rank_hint = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", rank_hint))
    device = torch.device("cpu")
    if args.backend == "nccl":
        if not torch.cuda.is_available():
            raise RuntimeError("NCCL overlap benchmark requires CUDA")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        torch.cuda.reset_peak_memory_stats(device)
        dist.init_process_group(args.backend, device_id=device)
    else:
        dist.init_process_group(args.backend)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", world_size))
    try:
        circuit = build_workload(args.n_wires, args.layers, device)
        for _ in range(args.warmup):
            _run_once(
                circuit,
                device=device,
                exchange_buffer_bytes=args.exchange_buffer_bytes,
                pipeline=False,
            )
            _run_once(
                circuit,
                device=device,
                exchange_buffer_bytes=args.exchange_buffer_bytes,
                pipeline=True,
            )
        profiles: dict[str, list[dict[str, Any]]] = {}
        if args.profile:
            trace_root = args.profile_output_dir
            profiles["synchronous"] = _profile_once(
                circuit,
                device=device,
                exchange_buffer_bytes=args.exchange_buffer_bytes,
                pipeline=False,
                trace_path=(
                    trace_root / f"rank-{rank}-synchronous.json"
                    if trace_root is not None
                    else None
                ),
            )
            profiles["pipeline"] = _profile_once(
                circuit,
                device=device,
                exchange_buffer_bytes=args.exchange_buffer_bytes,
                pipeline=True,
                trace_path=(
                    trace_root / f"rank-{rank}-pipeline.json"
                    if trace_root is not None
                    else None
                ),
            )
        pipeline_host: list[float] = []
        synchronous_host: list[float] = []
        pipeline_device: list[float] = []
        synchronous_device: list[float] = []
        pipeline_result = synchronous_result = None
        for index in range(args.repetitions):
            order = (False, True) if index % 2 == 0 else (True, False)
            for enabled in order:
                result, host_seconds, device_seconds = _run_once(
                    circuit,
                    device=device,
                    exchange_buffer_bytes=args.exchange_buffer_bytes,
                    pipeline=enabled,
                )
                if enabled:
                    pipeline_result = result
                    pipeline_host.append(host_seconds)
                    if device_seconds is not None:
                        pipeline_device.append(device_seconds)
                else:
                    synchronous_result = result
                    synchronous_host.append(host_seconds)
                    if device_seconds is not None:
                        synchronous_device.append(device_seconds)
        assert pipeline_result is not None and synchronous_result is not None
        max_error = float(
            torch.max(
                torch.abs(
                    pipeline_result.shard_state.amplitudes
                    - synchronous_result.shard_state.amplitudes
                )
            ).item()
        )
        local = {
            "rank": rank,
            "device": str(device),
            "pipeline_host": pipeline_host,
            "synchronous_host": synchronous_host,
            "pipeline_device": pipeline_device,
            "synchronous_device": synchronous_device,
            "max_error": max_error,
            "rank_ownership": pipeline_result.shard_state.summary(),
            "peak_memory_bytes": (
                int(torch.cuda.max_memory_allocated(device))
                if device.type == "cuda"
                else 0
            ),
            "profiles": profiles,
        }
        gathered: list[dict[str, Any] | None] = [None] * world_size
        dist.all_gather_object(gathered, local)
        if rank != 0:
            return None
        rows = [row for row in gathered if row is not None]
        pipeline_samples = _aggregate_rank_samples(rows, "pipeline_host")
        synchronous_samples = _aggregate_rank_samples(rows, "synchronous_host")
        pipeline_timing = _timing(pipeline_samples)
        synchronous_timing = _timing(synchronous_samples)
        workload = {
            "n_wires": args.n_wires,
            "layers": args.layers,
            "gate_count": len(circuit),
            "dtype": "complex64",
            "exchange_buffer_bytes": args.exchange_buffer_bytes,
        }
        workload_hash = hashlib.sha256(
            json.dumps(workload, sort_keys=True).encode("utf-8")
        ).hexdigest()
        payload: dict[str, Any] = {
            "schema_version": SCHEMA,
            "benchmark": "statevector_exchange_overlap",
            "artifact_class": "measured_development_run",
            "claim_evidence_type": "accelerator_development_performance"
            if args.backend == "nccl"
            else "development_semantics",
            "backend": args.backend,
            "distribution_semantics": "sharded_across_ranks",
            "world_size": world_size,
            "local_world_size": local_world_size,
            "node_count": max(
                1, (world_size + local_world_size - 1) // local_world_size
            ),
            "scalability_claim_allowed": False,
            "release_gate_allowed": False,
            "accelerator_overlap_measured": args.backend == "nccl",
            "workload": {**workload, "sha256": workload_hash},
            "environment": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "devices": [row["device"] for row in rows],
                "cuda_device_names": (
                    [
                        torch.cuda.get_device_name(index)
                        for index in range(torch.cuda.device_count())
                    ]
                    if torch.cuda.is_available()
                    else []
                ),
            },
            "pipeline": pipeline_timing,
            "synchronous": synchronous_timing,
            "host_speedup": (
                synchronous_timing["median_seconds"] / pipeline_timing["median_seconds"]
            ),
            "host_paired_improvement": _paired_improvement(
                pipeline_samples, synchronous_samples
            ),
            "correctness": {
                "passed": max(float(row["max_error"]) for row in rows) <= 2e-5,
                "max_abs_error": max(float(row["max_error"]) for row in rows),
                "absolute_tolerance": 2e-5,
            },
            "runtime": {
                "pipeline_prefetch_count": pipeline_result.exchange_pipeline_prefetch_count,
                "pipelined_gate_count": pipeline_result.exchange_pipelined_gate_count,
                "subgroup_pipelined_gate_count": pipeline_result.exchange_subgroup_pipelined_gate_count,
                "communication_count": pipeline_result.communication_count,
                "communication_bytes": pipeline_result.communication_bytes,
                "workspace_allocations": pipeline_result.exchange_workspace_allocation_count,
                "workspace_reuses": pipeline_result.exchange_workspace_reuse_count,
            },
            "rank_ownership": [row["rank_ownership"] for row in rows],
            "peak_memory_bytes": max(int(row["peak_memory_bytes"]) for row in rows),
        }
        if args.profile:
            payload["operator_profiles"] = [
                {"rank": row["rank"], **row["profiles"]} for row in rows
            ]
        if pipeline_device:
            pipeline_device_samples = _aggregate_rank_samples(rows, "pipeline_device")
            synchronous_device_samples = _aggregate_rank_samples(
                rows, "synchronous_device"
            )
            payload["device_timing"] = {
                "method": "cuda_events_with_device_synchronization",
                "pipeline": _timing(pipeline_device_samples),
                "synchronous": _timing(synchronous_device_samples),
                "paired_improvement": _paired_improvement(
                    pipeline_device_samples, synchronous_device_samples
                ),
            }
        decision_source = (
            payload.get("device_timing", {}).get("paired_improvement")
            or payload["host_paired_improvement"]
        )
        payload["overlap_benefit_demonstrated"] = bool(
            decision_source["benefit_demonstrated"]
        )
        payload["pipeline_recommendation"] = (
            "enable_for_measured_workload"
            if payload["overlap_benefit_demonstrated"]
            else "keep_disabled_pending_measured_benefit"
        )
        payload["validation_blockers"] = validate_overlap_payload(payload)
        return payload
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("gloo", "nccl"), default="gloo")
    parser.add_argument("--n-wires", type=int, default=20)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--exchange-buffer-bytes", type=int, default=4 * 1024 * 1024)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-output-dir", type=Path)
    args = parser.parse_args()
    payload = run(args)
    if payload is None:
        return
    encoded = json.dumps(payload, indent=2, sort_keys=True)
    print(encoded)
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(encoded + "\n", encoding="utf-8")
    if payload["validation_blockers"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
