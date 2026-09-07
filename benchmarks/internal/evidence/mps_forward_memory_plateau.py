#!/usr/bin/env python3
"""Certify bounded layer-local memory for site-sharded MPS forward execution."""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import socket
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.distributed as dist

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.internal.mps_hamiltonian_identification.core import (
    build_variable_time_batched_trotter_circuit,
    make_probes,
    smooth_couplings,
)
from flagquantum.runtime.backends.mps.forward import (
    execute_torch_distributed_mps_forward,
)
from flagquantum.runtime.backends.mps.factorization import FactorizationWorkspacePolicy
from flagquantum.runtime.backends.mps.reverse import site_sharded_z_zz_observations
from flagquantum.simulation.mps_site_kernels import (
    reset_site_kernel_stats,
    site_kernel_stats,
)
from flagquantum.simulation.mps.factorization import (
    mps_svd_fallback_stats,
    reset_mps_svd_fallback_stats,
)


SCHEMA = "flagquantum.issue107.mps_forward_memory_plateau.v1"


def _parse_depths(value: str) -> tuple[int, ...]:
    depths = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not depths or any(depth <= 0 for depth in depths):
        raise argparse.ArgumentTypeError(
            "depths must be positive comma-separated integers"
        )
    return depths


def _slope(values: Sequence[int]) -> float:
    if len(values) < 2:
        return 0.0
    center = (len(values) - 1) / 2.0
    denominator = sum((index - center) ** 2 for index in range(len(values)))
    return (
        sum(
            (index - center) * (float(value) - sum(values) / len(values))
            for index, value in enumerate(values)
        )
        / denominator
    )


def audit_plateau(
    rank_records: Sequence[dict[str, Any]],
    *,
    target_depth: int,
    memory_budget_bytes: int,
    maximum_slope_fraction_per_layer: float = 0.01,
) -> dict[str, Any]:
    """Return a fail-closed, JSON-safe layer-drain plateau decision."""

    slope_limit = float(memory_budget_bytes) * maximum_slope_fraction_per_layer
    blockers: list[str] = []
    rank_audits = []
    for rank_record in rank_records:
        depth_audits = []
        for run in rank_record["runs"]:
            records = run["layer_records"]
            if not records or any(
                item["remaining_precomputed_entries"] != 0 for item in records
            ):
                blockers.append(
                    f"rank_{rank_record['rank']}_depth_{run['depth']}_layer_cache_not_drained"
                )
            tail = records[-5:]
            allocated = [int(item["allocated_memory_bytes"] or 0) for item in tail]
            residual = [
                max(
                    0,
                    int(item["allocated_memory_bytes"] or 0)
                    - int(item["local_tensor_bytes"]),
                )
                for item in tail
            ]
            allocated_slope = _slope(allocated)
            residual_slope = _slope(residual)
            depth_audits.append(
                {
                    "depth": run["depth"],
                    "tail_layer_count": len(tail),
                    "allocated_slope_bytes_per_layer": allocated_slope,
                    "non_state_residual_slope_bytes_per_layer": residual_slope,
                    "slope_limit_bytes_per_layer": slope_limit,
                }
            )
            if max(0.0, residual_slope) >= slope_limit:
                blockers.append(
                    f"rank_{rank_record['rank']}_depth_{run['depth']}_residual_memory_growth"
                )
            if run["depth"] == target_depth:
                if max(0.0, allocated_slope) >= slope_limit:
                    blockers.append(
                        f"rank_{rank_record['rank']}_target_allocated_memory_growth"
                    )
                if run["peak_allocated_memory_bytes"] > memory_budget_bytes:
                    blockers.append(
                        f"rank_{rank_record['rank']}_target_peak_memory_exceeded"
                    )
        if target_depth not in {run["depth"] for run in rank_record["runs"]}:
            blockers.append(f"rank_{rank_record['rank']}_target_depth_missing")
        if any(not run["layer_cache_empty_at_return"] for run in rank_record["runs"]):
            blockers.append(f"rank_{rank_record['rank']}_return_cache_not_empty")
        rank_audits.append({"rank": rank_record["rank"], "depths": depth_audits})
    return {
        "passed": not blockers,
        "target_depth": target_depth,
        "memory_budget_bytes": memory_budget_bytes,
        "maximum_slope_fraction_per_layer": maximum_slope_fraction_per_layer,
        "slope_limit_bytes_per_layer": slope_limit,
        "rank_audits": rank_audits,
        "blockers": tuple(sorted(set(blockers))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--depths", type=_parse_depths, default=(1, 2, 4, 8, 12, 20))
    parser.add_argument("--n-wires", type=int, default=1024)
    parser.add_argument("--max-bond", type=int, default=1024)
    parser.add_argument("--cutoff", type=float, default=0.0)
    parser.add_argument("--discarded-weight-tolerance", type=float, default=0.1)
    parser.add_argument("--observation-stride", type=int, default=64)
    parser.add_argument("--dt", type=float, default=0.08)
    parser.add_argument("--memory-budget-gib", type=float, default=38.0)
    parser.add_argument("--heartbeat-every-layers", type=int, default=1)
    parser.add_argument("--compile-observables", action="store_true")
    parser.add_argument(
        "--compile-site-kernels",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--factorization-minimum-headroom-gib", type=float, default=2.0)
    parser.add_argument("--factorization-maximum-chunk-size", type=int, default=8)
    parser.add_argument(
        "--factorization-high-bond-maximum-chunk-size", type=int, default=1
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        args.n_wires < 2
        or args.max_bond < 1
        or args.heartbeat_every_layers < 1
        or args.factorization_minimum_headroom_gib < 0
        or args.factorization_maximum_chunk_size < 1
        or args.factorization_high_bond_maximum_chunk_size < 1
    ):
        raise SystemExit("n-wires >= 2, max-bond >= 1, and heartbeat interval >= 1")

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    reset_site_kernel_stats()
    reset_mps_svd_fallback_stats()
    try:
        coupling, field = smooth_couplings(args.n_wires, device=device)
        observation_sites = tuple(range(0, args.n_wires, args.observation_stride))
        target_depth = max(args.depths)
        probes = make_probes(
            args.n_wires, n_initial_states=1, time_steps=(target_depth,), seed=2026
        )
        circuit = build_variable_time_batched_trotter_circuit(
            coupling, field, probes, dt=args.dt, device=device
        )
        torch.cuda.reset_peak_memory_stats(device)
        allocator_before = torch.cuda.memory_stats(device)
        started = time.perf_counter()
        layer_records: list[dict[str, Any]] = []

        def heartbeat(record: dict[str, Any]) -> None:
            measured = {
                **record,
                "elapsed_seconds": time.perf_counter() - started,
            }
            layer_records.append(measured)
            if (
                rank == 0
                and record["layer_sequence"] % args.heartbeat_every_layers == 0
            ):
                print(
                    json.dumps(
                        {
                            "event": "issue107_layer_drain",
                            "rank": rank,
                            "target_depth": target_depth,
                            **measured,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

        with torch.no_grad():
            result = execute_torch_distributed_mps_forward(
                circuit,
                device=device,
                max_bond=args.max_bond,
                cutoff=args.cutoff,
                rebalance_threshold=float("inf"),
                global_error_budget=args.discarded_weight_tolerance,
                error_budget_policy="enforce",
                truncation_gradient_policy="approximate",
                compile_site_kernels=args.compile_site_kernels,
                layer_lifecycle_callback=heartbeat,
                factorization_workspace_policy=FactorizationWorkspacePolicy(
                    memory_budget_bytes=int(args.memory_budget_gib * (1 << 30)),
                    minimum_headroom_bytes=int(
                        args.factorization_minimum_headroom_gib * (1 << 30)
                    ),
                    maximum_chunk_size=args.factorization_maximum_chunk_size,
                    high_bond_maximum_chunk_size=(
                        args.factorization_high_bond_maximum_chunk_size
                    ),
                    policy_id="issue108_workspace_calibrated_v1",
                ),
            )
            observations = site_sharded_z_zz_observations(
                result.shard_state,
                observation_sites,
                compiled=args.compile_observables,
            )
        torch.cuda.synchronize(device)
        summary = result.summary()
        factorization_records = list(summary["factorization_workspace"]["records"])
        target_peak = int(torch.cuda.max_memory_allocated(device))
        rank_runs = []
        for depth in args.depths:
            prefix = layer_records[: 3 * depth]
            if len(prefix) != 3 * depth:
                raise RuntimeError(
                    f"depth {depth} expected {3 * depth} drained layers, got {len(prefix)}"
                )
            rank_runs.append(
                {
                    "depth": depth,
                    "measurement_semantics": "prefix_of_single_target_depth_execution",
                    "seconds": prefix[-1]["elapsed_seconds"],
                    "layer_cache_empty_at_return": all(
                        item["remaining_precomputed_entries"] == 0 for item in prefix
                    ),
                    "layer_records": prefix,
                    "peak_allocated_memory_bytes": (
                        target_peak
                        if depth == target_depth
                        else max(
                            int(item["peak_allocated_memory_bytes"] or 0)
                            for item in prefix
                        )
                    ),
                    "final_allocated_memory_bytes": int(
                        prefix[-1]["allocated_memory_bytes"] or 0
                    ),
                    "local_tensor_bytes": int(prefix[-1]["local_tensor_bytes"]),
                    "truncation_error": (
                        summary["truncation_error"] if depth == target_depth else None
                    ),
                    "observation_shape": (
                        tuple(observations.shape) if depth == target_depth else None
                    ),
                    "full_mps_reconstruction_count": 0,
                }
            )
        del observations, result, summary, circuit, probes
        gc.collect()
        torch.cuda.synchronize(device)
        cleanup_allocated = int(torch.cuda.memory_allocated(device))
        allocator_after = torch.cuda.memory_stats(device)
        for run in rank_runs:
            run["cleanup_allocated_memory_bytes"] = cleanup_allocated
        print(
            json.dumps(
                {
                    "event": "issue107_target_complete",
                    "rank": rank,
                    "target_depth": target_depth,
                    "peak_allocated_memory_bytes": target_peak,
                    "cleanup_allocated_memory_bytes": cleanup_allocated,
                },
                sort_keys=True,
            ),
            flush=True,
        )

        planned_factorizations = [
            record
            for record in factorization_records
            if "selected_chunk_size" in record
        ]
        rank_record = {
            "rank": rank,
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device),
            "runs": rank_runs,
            "site_kernel_stats": site_kernel_stats(),
            "svd_fallback_stats": mps_svd_fallback_stats(),
            "factorization_records": factorization_records,
            "factorization_summary": {
                "decision_count": len(factorization_records),
                "selected_chunk_histogram": {
                    str(size): sum(
                        int(record["selected_chunk_size"]) == size
                        for record in planned_factorizations
                    )
                    for size in sorted(
                        {
                            int(record["selected_chunk_size"])
                            for record in planned_factorizations
                        }
                    )
                },
                "downshift_count": sum(
                    bool(record["downshifted"]) for record in planned_factorizations
                ),
                "minimum_observed_free_bytes": min(
                    (
                        int(record["memory_snapshot"]["free_bytes"])
                        for record in planned_factorizations
                    ),
                    default=int(torch.cuda.mem_get_info(device)[0]),
                ),
                "maximum_selected_working_set_bytes": max(
                    (
                        int(record["selected_working_set_bytes"])
                        for record in planned_factorizations
                    ),
                    default=0,
                ),
                "minimum_headroom_bytes": int(
                    args.factorization_minimum_headroom_gib * (1 << 30)
                ),
                "allocator_retry_count": int(
                    allocator_after.get("num_alloc_retries", 0)
                    - allocator_before.get("num_alloc_retries", 0)
                ),
                "allocator_oom_count": int(
                    allocator_after.get("num_ooms", 0)
                    - allocator_before.get("num_ooms", 0)
                ),
            },
        }
        gathered = [None] * world if rank == 0 else None
        dist.gather_object(rank_record, gathered, dst=0)
        if rank == 0:
            assert gathered is not None
            memory_budget_bytes = int(args.memory_budget_gib * (1 << 30))
            audit = audit_plateau(
                gathered,
                target_depth=max(args.depths),
                memory_budget_bytes=memory_budget_bytes,
            )
            payload = {
                "schema": SCHEMA,
                "status": "passed" if audit["passed"] else "failed",
                "workload": {
                    "n_wires": args.n_wires,
                    "depths": args.depths,
                    "max_bond": args.max_bond,
                    "cutoff": args.cutoff,
                    "discarded_weight_tolerance": args.discarded_weight_tolerance,
                    "observation_stride": args.observation_stride,
                    "dtype": "complex64",
                    "execution_strategy": "single_max_depth_with_prefix_checkpoints",
                },
                "environment": {
                    "hostname": socket.gethostname(),
                    "platform": platform.platform(),
                    "torch": torch.__version__,
                    "cuda": torch.version.cuda,
                    "world_size": world,
                    "local_world_size": int(os.environ.get("LOCAL_WORLD_SIZE", world)),
                    "node_count": 1,
                    "compile_cache_dir": os.environ.get("TORCHINDUCTOR_CACHE_DIR"),
                },
                "distribution_semantics": "sharded_across_ranks",
                "full_mps_reconstruction_count": 0,
                "rank_records": gathered,
                "audit": audit,
                "scalability_claim_allowed": False,
                "blockers": audit["blockers"],
            }
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            if not audit["passed"]:
                raise RuntimeError(
                    f"ISSUE-107 plateau audit failed: {audit['blockers']}"
                )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
