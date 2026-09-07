#!/usr/bin/env python3
"""Calibrate owner-local bond-1024 RXX microbatch sizes on every GPU rank."""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flagquantum.simulation.mps.factorization import (
    _split_pair_matrix,
    _split_pair_matrix_bucket,
)
from flagquantum.simulation.mps.models import MPSConfig
from flagquantum.simulation.mps.site_kernels import apply_rxx_contraction_bucket


SCHEMA = "flagquantum.issue108.factorization_calibration.v1"


def _run_policy(
    lefts: list[torch.Tensor],
    rights: list[torch.Tensor],
    matrices: list[torch.Tensor],
    *,
    chunk_size: int,
    max_bond: int,
    strategy: str,
) -> float:
    started = time.perf_counter()
    config = MPSConfig(max_bond=max_bond, cutoff=0.0)
    for start in range(0, len(lefts), chunk_size):
        packed_left = torch.stack(lefts[start : start + chunk_size])
        packed_right = torch.stack(rights[start : start + chunk_size])
        packed_matrix = torch.stack(matrices[start : start + chunk_size])
        pairs = apply_rxx_contraction_bucket(
            packed_left, packed_right, packed_matrix, compiled=False
        )
        if strategy in {"batched", "exact_gesvd"}:
            outputs = _split_pair_matrix_bucket(
                pairs,
                left_dim=packed_left.shape[2],
                right_dim=packed_right.shape[-1],
                config=config,
                svd_driver="gesvd" if strategy == "exact_gesvd" else None,
            )
        elif strategy == "streams":
            producer = torch.cuda.current_stream(packed_left.device)
            streams = [torch.cuda.Stream(device=packed_left.device) for _ in pairs]
            outputs_list = [None] * len(pairs)
            for index, stream in enumerate(streams):
                stream.wait_stream(producer)
                with torch.cuda.stream(stream):
                    outputs_list[index] = _split_pair_matrix(
                        pairs[index],
                        left_dim=packed_left.shape[2],
                        right_dim=packed_right.shape[-1],
                        config=config,
                    )
            for stream in streams:
                producer.wait_stream(stream)
            outputs = tuple(outputs_list)
        elif strategy == "serial":
            outputs = tuple(
                _split_pair_matrix(
                    pair,
                    left_dim=packed_left.shape[2],
                    right_dim=packed_right.shape[-1],
                    config=config,
                )
                for pair in pairs
            )
        else:  # pragma: no cover - CLI policies are fixed below
            raise ValueError(f"unknown factorization strategy {strategy!r}")
        # Force all solver work to complete before workspace lifetime ends.
        torch.cuda.synchronize(packed_left.device)
        del outputs, pairs, packed_left, packed_right, packed_matrix
    return time.perf_counter() - started


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bond", type=int, default=1024)
    parser.add_argument("--items", type=int, default=4)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--memory-budget-gib", type=float, default=38.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.bond, args.items, args.samples) < 1 or args.warmups < 0:
        raise SystemExit("bond/items/samples must be positive and warmups non-negative")

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    try:
        generator = torch.Generator(device=device).manual_seed(108)
        scale = args.bond**-0.5
        lefts = [
            torch.randn(
                (1, args.bond, 2, args.bond),
                dtype=torch.complex64,
                device=device,
                generator=generator,
            )
            * scale
            for _ in range(args.items)
        ]
        rights = [item.clone() for item in lefts]
        matrices = [
            torch.eye(4, dtype=torch.complex64, device=device).reshape(1, 4, 4)
            for _ in range(args.items)
        ]
        policies = []
        for strategy, chunk_size in (
            ("serial", 1),
            ("exact_gesvd", 1),
            ("batched", 2),
            ("streams", 2),
        ):
            for _ in range(args.warmups):
                with torch.no_grad():
                    _run_policy(
                        lefts,
                        rights,
                        matrices,
                        chunk_size=chunk_size,
                        max_bond=args.bond,
                        strategy=strategy,
                    )
            samples = []
            peaks = []
            retries_before = int(
                torch.cuda.memory_stats(device).get("num_alloc_retries", 0)
            )
            ooms_before = int(torch.cuda.memory_stats(device).get("num_ooms", 0))
            for _ in range(args.samples):
                torch.cuda.reset_peak_memory_stats(device)
                with torch.no_grad():
                    samples.append(
                        _run_policy(
                            lefts,
                            rights,
                            matrices,
                            chunk_size=chunk_size,
                            max_bond=args.bond,
                            strategy=strategy,
                        )
                    )
                peaks.append(int(torch.cuda.max_memory_allocated(device)))
            memory_stats = torch.cuda.memory_stats(device)
            policies.append(
                {
                    "chunk_size": chunk_size,
                    "strategy": strategy,
                    "warmups": args.warmups,
                    "samples_seconds": samples,
                    "mean_seconds": statistics.fmean(samples),
                    "median_seconds": statistics.median(samples),
                    "peak_allocated_memory_bytes": max(peaks),
                    "allocator_retry_count": int(
                        memory_stats.get("num_alloc_retries", 0) - retries_before
                    ),
                    "allocator_oom_count": int(
                        memory_stats.get("num_ooms", 0) - ooms_before
                    ),
                }
            )
        serial = policies[0]
        memory_budget_bytes = int(args.memory_budget_gib * (1 << 30))
        eligible = [
            policy
            for policy in policies
            if policy["mean_seconds"] <= serial["mean_seconds"] * 0.99
            and policy["peak_allocated_memory_bytes"] <= memory_budget_bytes
            and policy["allocator_retry_count"] == 0
            and policy["allocator_oom_count"] == 0
        ]
        selected_policy = min(
            eligible, key=lambda item: item["mean_seconds"], default=serial
        )
        rank_record = {
            "rank": rank,
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device),
            "policies": policies,
            "selected_chunk_size": selected_policy["chunk_size"],
            "selected_strategy": selected_policy["strategy"],
            "selection_reason": (
                "at_least_one_percent_faster_within_memory_budget"
                if selected_policy is not serial
                else "no_parallel_strategy_measured_materially_faster"
            ),
        }
        gathered = [None] * world if rank == 0 else None
        dist.gather_object(rank_record, gathered, dst=0)
        if rank == 0:
            assert gathered is not None
            selected_strategies = {
                str(record["selected_strategy"]) for record in gathered
            }
            selected_strategy = (
                next(iter(selected_strategies))
                if len(selected_strategies) == 1
                else "serial"
            )
            selected = 1 if selected_strategy == "serial" else 2
            blockers = []
            if any(
                policy["allocator_retry_count"] or policy["allocator_oom_count"]
                for record in gathered
                for policy in record["policies"]
            ):
                blockers.append("allocator_retry_or_oom")
            payload = {
                "schema": SCHEMA,
                "status": "passed" if not blockers else "failed",
                "workload": {
                    "bond": args.bond,
                    "items_per_rank": args.items,
                    "dtype": "complex64",
                    "warmups": args.warmups,
                    "samples": args.samples,
                },
                "environment": {
                    "hostname": socket.gethostname(),
                    "platform": platform.platform(),
                    "torch": torch.__version__,
                    "cuda": torch.version.cuda,
                    "world_size": world,
                },
                "memory_budget_bytes": memory_budget_bytes,
                "rank_records": gathered,
                "selected_high_bond_chunk_size": selected,
                "selected_high_bond_strategy": selected_strategy,
                "selection_semantics": "all_rank_conservative_minimum",
                "blockers": blockers,
            }
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            if blockers:
                raise RuntimeError(f"ISSUE-108 calibration failed: {blockers}")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
