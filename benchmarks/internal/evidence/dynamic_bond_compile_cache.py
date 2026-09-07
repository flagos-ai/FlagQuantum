"""ISSUE-105 bounded dynamic-bond compile-cache soak."""

from __future__ import annotations

import argparse
import json
import os
import socket
import time
from pathlib import Path

import torch
import torch.distributed as dist

from flagquantum.simulation.mps.site_kernels import (
    SiteKernelBucket,
    SiteKernelCachePolicy,
    apply_ry_bucket,
    configure_site_kernel_cache,
    prewarm_site_kernel_buckets,
    require_warm_step_regression,
    reset_site_kernel_stats,
    site_kernel_cache_events,
    site_kernel_stats,
)


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    dist.barrier()


def inputs(bond: int, device: torch.device, dtype: torch.dtype):
    generator = torch.Generator(device=device).manual_seed(105 + bond)
    tensors = torch.randn(4, 1, bond, 2, bond, device=device, dtype=dtype, generator=generator)
    matrices = torch.eye(2, device=device, dtype=dtype).expand(4, 1, 2, 2)
    return tensors, matrices


def sample(tensors, matrices, *, compiled: bool, device: torch.device) -> tuple[float, torch.Tensor]:
    sync(device)
    started = time.perf_counter()
    output = apply_ry_bucket(tensors, matrices, compiled=compiled)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    return elapsed, output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--maximum-warm-ratio", type=float, default=1.05)
    args = parser.parse_args()
    if args.steps != 100:
        raise ValueError("ISSUE-105 acceptance evidence requires exactly 100 warm steps")

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    dtype = torch.complex64
    hot_bonds = (32, 64, 128)
    policy = SiteKernelCachePolicy(
        bond_buckets=(1, 2, 4, 8, 16, 32, 64, 128),
        max_entries=3,
        max_accounted_input_bytes=64 * 1024 * 1024,
        mode="default",
        policy_id="issue105_a100_v1",
    )
    configure_site_kernel_cache(policy)
    reset_site_kernel_stats(clear_cache=True)

    prewarm_started = time.perf_counter()
    prewarm_events = prewarm_site_kernel_buckets(
        tuple(SiteKernelBucket("ry", bucket_size=4, left_bond=bond, right_bond=bond) for bond in hot_bonds),
        device=device,
        dtype=dtype,
    )
    sync(device)
    prewarm_seconds = time.perf_counter() - prewarm_started
    cold_stats = site_kernel_stats()
    reset_site_kernel_stats()

    prepared = {bond: inputs(bond, device, dtype) for bond in hot_bonds}
    eager_seconds = []
    for step in range(args.steps):
        bond = hot_bonds[step % len(hot_bonds)]
        elapsed, _ = sample(*prepared[bond], compiled=False, device=device)
        eager_seconds.append(elapsed)

    warm_seconds, memory_allocated, max_error = [], [], 0.0
    for step in range(args.steps):
        bond = hot_bonds[step % len(hot_bonds)]
        tensors, matrices = prepared[bond]
        elapsed, actual = sample(tensors, matrices, compiled=True, device=device)
        expected = apply_ry_bucket(tensors, matrices, compiled=False)
        max_error = max(max_error, float(torch.max(torch.abs(actual - expected))))
        warm_seconds.append(elapsed)
        memory_allocated.append(
            int(torch.cuda.memory_allocated(device)) if device.type == "cuda" else 0
        )

    entries_before_fallback = int(site_kernel_stats()["cache_entries"])
    rare = inputs(3, device, dtype)
    _, rare_actual = sample(*rare, compiled=True, device=device)
    rare_expected = apply_ry_bucket(*rare, compiled=False)
    rare_error = float(torch.max(torch.abs(rare_actual - rare_expected)))
    warm_stats = site_kernel_stats()
    events = site_kernel_cache_events()
    regression = require_warm_step_regression(
        warm_seconds, eager_seconds, maximum_ratio=args.maximum_warm_ratio
    )

    record = {
        "rank": rank,
        "hostname": socket.gethostname(),
        "device": str(device),
        "prewarm_seconds": prewarm_seconds,
        "prewarm_events": prewarm_events,
        "cold_stats": cold_stats,
        "warm_stats": warm_stats,
        "warm_events": events,
        "eager_seconds": eager_seconds,
        "warm_seconds": warm_seconds,
        "memory_allocated": memory_allocated,
        "max_error": max(max_error, rare_error),
        "entries_before_fallback": entries_before_fallback,
        "entries_after_fallback": int(warm_stats["cache_entries"]),
        "regression_gate": regression,
    }
    records = [None] * world
    dist.all_gather_object(records, record)
    if rank == 0:
        payload = {
            "schema": "flagquantum.issue105.dynamic_bond_compile_cache.v1",
            "world_size": world,
            "local_world_size": int(os.environ.get("LOCAL_WORLD_SIZE", world)),
            "node_count": len({item["hostname"] for item in records}),
            "distribution_semantics": "sharded_across_ranks",
            "scalability_claim_allowed": False,
            "workload": {
                "steps": args.steps,
                "hot_bond_buckets": list(hot_bonds),
                "rank_local_sites_per_bucket": 4,
                "dtype": str(dtype),
                "cache_policy": {
                    "policy_id": policy.policy_id,
                    "max_entries": policy.max_entries,
                    "max_accounted_input_bytes": policy.max_accounted_input_bytes,
                    "unsupported_shape": policy.unsupported_shape,
                },
            },
            "post_warmup_compile_count": max(item["warm_stats"]["cache_misses"] for item in records),
            "cache_entries_max": max(item["warm_stats"]["cache_entries"] for item in records),
            "cache_accounted_input_bytes_max": max(item["warm_stats"]["cache_accounted_input_bytes"] for item in records),
            "memory_growth_bytes_max": max(
                max(item["memory_allocated"]) - min(item["memory_allocated"])
                for item in records
            ),
            "max_correctness_error": max(item["max_error"] for item in records),
            "warm_regression_passed": all(item["regression_gate"]["passed"] for item in records),
            "maximum_warm_ratio": args.maximum_warm_ratio,
            "maximum_observed_warm_ratio": max(item["regression_gate"]["compiled_to_eager_ratio"] for item in records),
            "unsupported_fallbacks": sum(item["warm_stats"]["eager_fallbacks"] for item in records),
            "fallback_preserved_cache": all(item["entries_before_fallback"] == item["entries_after_fallback"] for item in records),
            "ranks": records,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
