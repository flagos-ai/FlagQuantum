#!/usr/bin/env python3
"""Reproducible local-GPU and distributed-shard performance baseline runner."""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import time
from pathlib import Path

import torch
import torch.distributed as dist

from flagquantum.runtime.observability.performance import (
    PerformanceRecord,
    PerformanceThresholds,
    calibrate_cost_model,
    evaluate_performance,
)


def _gpu_inventory() -> tuple[str, ...]:
    return tuple(
        torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
    )


def _commit() -> str:
    return subprocess.run(
        ("git", "rev-parse", "HEAD"), check=True, capture_output=True, text=True
    ).stdout.strip()


def _measure_local(
    warmup: int, repetitions: int, inner_loops: int
) -> PerformanceRecord:
    started = time.perf_counter()
    if not torch.cuda.is_available():
        raise RuntimeError("local GPU benchmark requires CUDA")
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    generator = torch.Generator(device=device).manual_seed(77)
    matrix = torch.randn((512, 512), generator=generator, device=device)
    matrix = matrix.to(torch.complex64) / (512**0.5)
    vector = torch.randn((512, 256), generator=generator, device=device).to(
        torch.complex64
    )
    for _ in range(warmup):
        vector = matrix @ vector
    torch.cuda.synchronize(device)
    initialization = time.perf_counter() - started
    torch.cuda.reset_peak_memory_stats(device)
    samples = []
    allocated = []
    reserved = []
    heartbeats = []
    last = time.perf_counter()
    for _ in range(repetitions):
        start = time.perf_counter()
        for _ in range(inner_loops):
            vector = matrix @ vector
        torch.cuda.synchronize(device)
        end = time.perf_counter()
        samples.append(end - start)
        heartbeats.append(end - last)
        last = end
        allocated.append(torch.cuda.memory_allocated(device))
        reserved.append(torch.cuda.memory_reserved(device))
    return PerformanceRecord(
        benchmark="issue077_local_gpu_kernel",
        layer="kernel",
        backend="pytorch",
        device=str(device),
        distribution_semantics="single_device_fast_path",
        world_size=1,
        warmup=warmup,
        repetitions=repetitions,
        samples_seconds=tuple(samples),
        initialization_seconds=initialization,
        teardown_seconds=0.0,
        peak_memory_allocated_bytes=torch.cuda.max_memory_allocated(device),
        peak_memory_reserved_bytes=torch.cuda.max_memory_reserved(device),
        allocated_memory_by_step=tuple(allocated),
        reserved_memory_by_step=tuple(reserved),
        useful_work_seconds=sum(samples),
        communication_seconds=0.0,
        idle_seconds=0.0,
        completed_work_units=repetitions * inner_loops,
        heartbeat_gaps_seconds=tuple(heartbeats),
        correctness_passed=bool(torch.isfinite(vector).all()),
    )


def _measure_jax_local(
    warmup: int, repetitions: int, inner_loops: int
) -> PerformanceRecord:
    import jax
    import jax.numpy as jnp

    device = jax.devices("gpu")[0]
    key = jax.random.key(77)
    matrix = jax.random.normal(key, (512, 512), dtype=jnp.float32).astype(jnp.complex64)
    matrix = matrix / (512**0.5)
    vector = jax.random.normal(key, (512, 256), dtype=jnp.float32).astype(jnp.complex64)
    started = time.perf_counter()
    for _ in range(warmup):
        vector = matrix @ vector
    vector.block_until_ready()
    initialization = time.perf_counter() - started
    samples = []
    heartbeats = []
    last = time.perf_counter()
    for _ in range(repetitions):
        start = time.perf_counter()
        for _ in range(inner_loops):
            vector = matrix @ vector
        vector.block_until_ready()
        end = time.perf_counter()
        samples.append(end - start)
        heartbeats.append(end - last)
        last = end
    memory = device.memory_stats() or {}
    allocated = int(memory.get("bytes_in_use", 0))
    reserved = int(memory.get("peak_bytes_in_use", allocated))
    return PerformanceRecord(
        benchmark="issue077_local_gpu_jax_kernel",
        layer="kernel",
        backend="jax",
        device=str(device),
        distribution_semantics="single_device_fast_path",
        world_size=1,
        warmup=warmup,
        repetitions=repetitions,
        samples_seconds=tuple(samples),
        initialization_seconds=initialization,
        teardown_seconds=0.0,
        peak_memory_allocated_bytes=allocated,
        peak_memory_reserved_bytes=reserved,
        allocated_memory_by_step=(allocated,) * repetitions,
        reserved_memory_by_step=(reserved,) * repetitions,
        useful_work_seconds=sum(samples),
        communication_seconds=0.0,
        idle_seconds=0.0,
        completed_work_units=repetitions * inner_loops,
        heartbeat_gaps_seconds=tuple(heartbeats),
        correctness_passed=bool(jnp.isfinite(vector).all()),
    )


def _measure_distributed(
    warmup: int, repetitions: int, inner_loops: int
) -> tuple[PerformanceRecord, dict[str, object]]:
    started = time.perf_counter()
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    shard_size = 1 << 18
    shard = torch.full((shard_size,), rank + 1, dtype=torch.float32, device=device)
    heartbeat = torch.zeros((), dtype=torch.float32, device=device)
    for _ in range(warmup):
        shard.mul_(1.000001).sub_(0.000001 * (rank + 1))
        heartbeat.copy_(shard.mean())
        dist.all_reduce(heartbeat)
    torch.cuda.synchronize(device)
    initialization = time.perf_counter() - started
    torch.cuda.reset_peak_memory_stats(device)
    samples = []
    communication = 0.0
    allocated = []
    reserved = []
    heartbeat_gaps = []
    last = time.perf_counter()
    for _ in range(repetitions):
        step_start = time.perf_counter()
        step_communication = 0.0
        for _ in range(inner_loops):
            shard.mul_(1.000001).sub_(0.000001 * (rank + 1))
            heartbeat.copy_(shard.mean())
            # NCCL work is asynchronous with respect to the host. Synchronize
            # both timing boundaries so this measures device completion rather
            # than Python enqueue latency.
            torch.cuda.synchronize(device)
            communication_start = time.perf_counter()
            dist.all_reduce(heartbeat)
            torch.cuda.synchronize(device)
            step_communication += time.perf_counter() - communication_start
        torch.cuda.synchronize(device)
        end = time.perf_counter()
        communication += step_communication
        samples.append(end - step_start)
        heartbeat_gaps.append(end - last)
        last = end
        allocated.append(torch.cuda.memory_allocated(device))
        reserved.append(torch.cuda.memory_reserved(device))
    expected = sum(range(1, world_size + 1))
    correct = abs(float(heartbeat.item()) - expected) < 0.01 * expected
    local = {
        "rank": rank,
        "local_rank": local_rank,
        "device": str(device),
        "owned_range": (rank * shard_size, (rank + 1) * shard_size),
        "samples": samples,
        "communication_seconds": communication,
        "allocated": allocated,
        "reserved": reserved,
        "peak_allocated": torch.cuda.max_memory_allocated(device),
        "peak_reserved": torch.cuda.max_memory_reserved(device),
        "correct": correct,
    }
    gathered: list[dict[str, object] | None] = [None] * world_size
    dist.all_gather_object(gathered, local)
    teardown_start = time.perf_counter()
    dist.destroy_process_group()
    teardown = time.perf_counter() - teardown_start
    rows = [item for item in gathered if item is not None]
    combined_samples = tuple(
        max(float(row["samples"][index]) for row in rows)  # type: ignore[index]
        for index in range(repetitions)
    )
    communication_total = max(float(row["communication_seconds"]) for row in rows)
    useful = sum(combined_samples)
    record = PerformanceRecord(
        benchmark=f"issue077_{world_size}gpu_state_shard_communication",
        layer="communication",
        backend="pytorch_nccl",
        device="cuda",
        distribution_semantics="sharded_across_ranks",
        world_size=world_size,
        warmup=warmup,
        repetitions=repetitions,
        samples_seconds=combined_samples,
        initialization_seconds=initialization,
        teardown_seconds=teardown,
        peak_memory_allocated_bytes=max(int(row["peak_allocated"]) for row in rows),
        peak_memory_reserved_bytes=max(int(row["peak_reserved"]) for row in rows),
        allocated_memory_by_step=tuple(
            max(int(row["allocated"][index]) for row in rows)  # type: ignore[index]
            for index in range(repetitions)
        ),
        reserved_memory_by_step=tuple(
            max(int(row["reserved"][index]) for row in rows)  # type: ignore[index]
            for index in range(repetitions)
        ),
        useful_work_seconds=max(useful - communication_total, 0.0),
        communication_seconds=communication_total,
        idle_seconds=0.0,
        completed_work_units=repetitions * inner_loops * world_size,
        heartbeat_gaps_seconds=combined_samples,
        correctness_passed=all(bool(row["correct"]) for row in rows),
        explanation="Distinct state segments are rank-owned; scalar all-reduce tracks communication progress.",
    )
    return record, {
        "rank_ownership": rows,
        "shard_elements_total": shard_size * world_size,
    }


def _measure_crossover(
    warmup: int, repetitions: int, inner_loops: int, global_elements: int
) -> tuple[PerformanceRecord, dict[str, object]]:
    started = time.perf_counter()
    dist.init_process_group("nccl")
    world_size = dist.get_world_size()
    if global_elements % world_size:
        raise ValueError("global elements must be divisible by world size")
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    local_elements = global_elements // world_size
    shard = torch.linspace(0.0, 1.0, local_elements, device=device)
    checksum = torch.zeros((), device=device)

    def step() -> None:
        nonlocal shard
        shard = torch.sin(shard * 1.000001 + 0.000001)
        checksum.copy_(shard.sum())
        if world_size > 1:
            dist.all_reduce(checksum)

    for _ in range(warmup):
        step()
    torch.cuda.synchronize(device)
    initialization = time.perf_counter() - started
    torch.cuda.reset_peak_memory_stats(device)
    samples, allocated, reserved = [], [], []
    communication = 0.0
    for _ in range(repetitions):
        dist.barrier()
        torch.cuda.synchronize(device)
        begin = time.perf_counter()
        for _ in range(inner_loops):
            if world_size > 1:
                torch.cuda.synchronize(device)
                communication_start = time.perf_counter()
            step()
            if world_size > 1:
                torch.cuda.synchronize(device)
                communication += time.perf_counter() - communication_start
        torch.cuda.synchronize(device)
        samples.append(time.perf_counter() - begin)
        allocated.append(torch.cuda.memory_allocated(device))
        reserved.append(torch.cuda.memory_reserved(device))
    local = {
        "samples": samples,
        "communication": communication,
        "allocated": allocated,
        "reserved": reserved,
        "peak_allocated": torch.cuda.max_memory_allocated(device),
        "peak_reserved": torch.cuda.max_memory_reserved(device),
        "correct": bool(torch.isfinite(checksum)),
    }
    gathered: list[dict[str, object] | None] = [None] * world_size
    dist.all_gather_object(gathered, local)
    rows = [row for row in gathered if row is not None]
    teardown_started = time.perf_counter()
    dist.destroy_process_group()
    combined = tuple(
        max(float(row["samples"][index]) for row in rows)  # type: ignore[index]
        for index in range(repetitions)
    )
    communication_total = max(float(row["communication"]) for row in rows)
    total = sum(combined)
    record = PerformanceRecord(
        benchmark=f"issue077_crossover_{world_size}gpu_{global_elements}",
        layer="kernel",
        backend="pytorch_nccl" if world_size > 1 else "pytorch",
        device="cuda",
        distribution_semantics="matched_global_workload",
        world_size=world_size,
        warmup=warmup,
        repetitions=repetitions,
        samples_seconds=combined,
        initialization_seconds=initialization,
        teardown_seconds=time.perf_counter() - teardown_started,
        peak_memory_allocated_bytes=max(int(row["peak_allocated"]) for row in rows),
        peak_memory_reserved_bytes=max(int(row["peak_reserved"]) for row in rows),
        allocated_memory_by_step=tuple(
            max(int(row["allocated"][index]) for row in rows)  # type: ignore[index]
            for index in range(repetitions)
        ),
        reserved_memory_by_step=tuple(
            max(int(row["reserved"][index]) for row in rows)  # type: ignore[index]
            for index in range(repetitions)
        ),
        useful_work_seconds=max(total - communication_total, 0.0),
        communication_seconds=communication_total,
        idle_seconds=0.0,
        completed_work_units=repetitions * inner_loops * global_elements,
        heartbeat_gaps_seconds=combined,
        correctness_passed=all(bool(row["correct"]) for row in rows),
        explanation="Identical global pointwise workload; only ownership and scalar synchronization differ.",
    )
    return record, {
        "global_elements": global_elements,
        "local_elements_per_rank": local_elements,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("local-gpu", "distributed-shard", "crossover"),
        required=True,
    )
    parser.add_argument("--backend", choices=("pytorch", "jax"), default="pytorch")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=12)
    parser.add_argument("--inner-loops", type=int, default=50)
    parser.add_argument("--global-elements", type=int, default=1 << 20)
    args = parser.parse_args()
    if args.mode == "local-gpu":
        record = (
            _measure_jax_local(args.warmup, args.repetitions, args.inner_loops)
            if args.backend == "jax"
            else _measure_local(args.warmup, args.repetitions, args.inner_loops)
        )
        extra: dict[str, object] = {}
        write = True
    elif args.mode == "distributed-shard":
        record, extra = _measure_distributed(
            args.warmup, args.repetitions, args.inner_loops
        )
        write = int(os.environ.get("RANK", "0")) == 0
    else:
        record, extra = _measure_crossover(
            args.warmup,
            args.repetitions,
            args.inner_loops,
            args.global_elements,
        )
        write = int(os.environ.get("RANK", "0")) == 0
    if write:
        thresholds = PerformanceThresholds(
            max_coefficient_of_variation=0.25 if record.world_size >= 8 else 0.10
        )
        payload = {
            **record.summary(),
            **extra,
            "artifact_class": "development_run",
            "benchmark_evidence_class": "non_release_performance_baseline",
            "non_release_evidence": True,
            "scalability_claim_allowed": False,
            "release_gate_allowed": False,
            "scalability_blockers": ["development performance baseline only"],
            "cost_model_calibration": calibrate_cost_model(record).__dict__,
            "performance_gate": evaluate_performance(
                record, thresholds=thresholds
            ).__dict__,
            "regression_thresholds": thresholds.__dict__,
            "commit": _commit(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu_inventory": _gpu_inventory(),
            "sample_median_seconds": statistics.median(record.samples_seconds),
            "scaling_classification": (
                "strong_scaling_matched_workload"
                if args.mode == "crossover"
                else
                "not_applicable_local"
                if record.world_size == 1
                else "weak_scaling_capacity"
            ),
            "workload_dimensions": {
                "matrix_shape": [512, 512] if args.mode == "local-gpu" else None,
                "shard_elements_per_rank": (
                    None if record.world_size == 1 else 1 << 18
                ),
                "global_shard_elements": extra.get("shard_elements_total"),
                "global_elements": extra.get("global_elements"),
                "inner_loops": args.inner_loops,
            },
            "crossover": {
                "status": "not_measured",
                "dimensions": [],
                "single_gpu_baseline": None,
                "distributed_selection_threshold": None,
                "uncertainty_margin": record.robust_coefficient_of_variation,
                "blockers": [
                    "matched_workload_single_and_multi_gpu_sweep_required"
                ],
            },
            "communication_timing_method": (
                "not_applicable"
                if record.world_size == 1
                else "device_synchronized_wall_clock"
            ),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
