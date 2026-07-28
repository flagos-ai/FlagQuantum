#!/usr/bin/env python3
"""Measure fixed-workload statevector strong scaling under torchrun.

Each invocation measures one world size. Artifacts are development evidence:
they retain rank-level timings, memory, communication and invariant checks but
cannot satisfy the ISSUE-044 production release gate by themselves.
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
from flagquantum.runtime.backends.statevector.forward import (  # noqa: E402
    execute_torch_distributed_statevector,
)

SCHEMA = "flagquantum.statevector.strong_scaling.v1"


def build_workload(n_wires: int, depth: int, device: torch.device) -> fq.Circuit:
    """Build a deterministic fixed circuit whose IR is world-size independent."""

    if n_wires < 6 or depth < 1:
        raise ValueError("n_wires >= 6 and depth >= 1 required")
    circuit = fq.Circuit(n_wires, device=device)
    for layer in range(depth):
        scale = float(layer + 1)
        a = layer % n_wires
        b = (layer * 5 + 3) % n_wires
        c = (layer * 7 + n_wires - 1) % n_wires
        if b == a:
            b = (b + 1) % n_wires
        if c in {a, b}:
            c = (c + 2) % n_wires
        circuit.h(a).rx(b, 0.013 * scale).rz(c, -0.017 * scale)
        circuit.cx(a, b).ry(c, 0.011 * scale).cx(b, c)
        circuit.phase(a, 0.007 * scale).rzz(a, c, -0.009 * scale)
    return circuit


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _global_indices(result: Any, device: torch.device) -> torch.Tensor:
    shard = result.shard_state
    if shard.global_indices.numel():
        return shard.global_indices
    local = torch.arange(shard.shard.local_amplitudes, dtype=torch.long, device=device)
    if result.plan.distribution == "qubit_address_sharded":
        return (local << len(result.plan.sharded_wires)) | shard.rank
    return local + shard.shard.amplitude_start


def _invariants(result: Any, device: torch.device) -> dict[str, Any]:
    amplitudes = result.shard_state.amplitudes
    probabilities = amplitudes.abs().square().sum(dim=0)
    indices = _global_indices(result, device)
    values = [probabilities.sum()]
    observed_wires = tuple(range(min(3, result.plan.n_wires)))
    for wire in observed_wires:
        bits = (indices >> (result.plan.n_wires - wire - 1)) & 1
        signs = 1 - 2 * bits.to(probabilities.dtype)
        values.append((probabilities * signs).sum())
    packed = torch.stack(values)
    if dist.get_world_size() > 1:
        dist.all_reduce(packed, op=dist.ReduceOp.SUM)
    return {
        "norm": float(packed[0]),
        "z_expectations": {
            str(wire): float(packed[index + 1])
            for index, wire in enumerate(observed_wires)
        },
    }


def _run_once(
    circuit: fq.Circuit,
    *,
    device: torch.device,
    exchange_buffer_bytes: int,
) -> tuple[Any, float, float]:
    dist.barrier()
    _sync(device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    started = time.perf_counter()
    result = execute_torch_distributed_statevector(
        circuit,
        device=device,
        exchange_buffer_bytes=exchange_buffer_bytes,
        pipeline_pair_exchange=True,
    )
    end.record()
    _sync(device)
    host_seconds = time.perf_counter() - started
    device_seconds = float(start.elapsed_time(end)) / 1000.0
    dist.barrier()
    return result, host_seconds, device_seconds


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


def validate_payload(payload: dict[str, Any]) -> tuple[str, ...]:
    blockers = []
    if payload.get("schema_version") != SCHEMA:
        blockers.append("invalid_schema")
    if payload.get("artifact_class") != "measured_development_run":
        blockers.append("invalid_artifact_class")
    if payload.get("scalability_claim_allowed") is not False:
        blockers.append("development_artifact_must_reject_scalability_claim")
    if payload.get("release_gate_allowed") is not False:
        blockers.append("development_artifact_must_reject_release")
    if len(payload.get("rank_timings", [])) != int(payload.get("world_size", 0)):
        blockers.append("incomplete_rank_timings")
    if len(payload.get("rank_peak_memory_bytes", [])) != int(
        payload.get("world_size", 0)
    ):
        blockers.append("incomplete_rank_memory")
    world_size = int(payload.get("world_size", 0))
    rows = payload.get("rank_timings", [])
    if {row.get("rank") for row in rows} != set(range(world_size)):
        blockers.append("invalid_rank_coverage")
    placements = payload.get("rank_placement", [])
    if len(placements) != world_size or any(
        not placement.get("hostname") for placement in placements
    ):
        blockers.append("incomplete_rank_placement")
    node_count = int(payload.get("node_count", 0))
    if placements and len({row["hostname"] for row in placements}) != node_count:
        blockers.append("node_count_does_not_match_placement")
    communication_tiers = payload.get("communication_tiers", {})
    if node_count > 1 and not (
        communication_tiers.get("route_classification") == "inter_node_collective"
        and int(communication_tiers.get("inter_node_logical_bytes_per_rank_max", 0))
        > 0
    ):
        blockers.append("missing_inter_node_communication_evidence")
    correctness = payload.get("correctness", {})
    if not correctness.get("passed"):
        blockers.append("correctness_invariants_failed")
    if int(payload.get("timing", {}).get("sample_count", 0)) < 3:
        blockers.append("insufficient_timing_samples")
    return tuple(blockers)


def run(args: argparse.Namespace) -> dict[str, Any] | None:
    if args.repetitions < 3 or args.warmup < 0:
        raise ValueError("repetitions >= 3 and warmup >= 0 required")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if not torch.cuda.is_available():
        raise RuntimeError("strong-scaling evidence requires CUDA")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", world_size))
    node_rank = int(os.environ.get("GROUP_RANK", os.environ.get("NODE_RANK", 0)))
    try:
        circuit = build_workload(args.n_wires, args.depth, device)
        workload_ir = circuit.to_ir().to_dict()
        workload_sha256 = hashlib.sha256(
            json.dumps(workload_ir, sort_keys=True, default=str).encode()
        ).hexdigest()
        for _ in range(args.warmup):
            _run_once(
                circuit,
                device=device,
                exchange_buffer_bytes=args.exchange_buffer_bytes,
            )
        torch.cuda.reset_peak_memory_stats(device)
        host_samples: list[float] = []
        device_samples: list[float] = []
        last_result = None
        for _ in range(args.repetitions):
            last_result, host_seconds, device_seconds = _run_once(
                circuit,
                device=device,
                exchange_buffer_bytes=args.exchange_buffer_bytes,
            )
            host_samples.append(host_seconds)
            device_samples.append(device_seconds)
        assert last_result is not None
        invariants = _invariants(last_result, device)
        rank_row = {
            "rank": rank,
            "local_rank": local_rank,
            "node_rank": node_rank,
            "hostname": platform.node(),
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device),
            "host_samples_seconds": host_samples,
            "device_samples_seconds": device_samples,
            "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
            "local_state_bytes": int(
                last_result.shard_state.amplitudes.numel()
                * last_result.shard_state.amplitudes.element_size()
            ),
            "communication_count": last_result.communication_count,
            "communication_bytes": last_result.communication_bytes,
            "executor_peak_scratch_bytes": last_result.peak_scratch_bytes,
            "exchange_workspace_reserved_bytes": (
                last_result.exchange_workspace_reserved_bytes
            ),
            "exchange_workspace_allocation_count": (
                last_result.exchange_workspace_allocation_count
            ),
            "exchange_workspace_reuse_count": (
                last_result.exchange_workspace_reuse_count
            ),
            "invariants": invariants,
        }
        gathered: list[dict[str, Any] | None] = [None] * world_size
        dist.all_gather_object(gathered, rank_row)
        if rank != 0:
            return None
        rows = [row for row in gathered if row is not None]
        hostnames = sorted({row["hostname"] for row in rows})
        node_count = len(hostnames)
        host = [
            max(row["host_samples_seconds"][index] for row in rows)
            for index in range(args.repetitions)
        ]
        device_times = [
            max(row["device_samples_seconds"][index] for row in rows)
            for index in range(args.repetitions)
        ]
        communication_bytes = max(row["communication_bytes"] for row in rows)
        state_bytes = sum(row["local_state_bytes"] for row in rows)
        communication_fraction = communication_bytes / max(
            1, communication_bytes + state_bytes
        )
        invariant_rows = [row["invariants"] for row in rows]
        norm_error = max(abs(row["norm"] - 1.0) for row in invariant_rows)
        rank_invariant_spread = max(
            abs(row["norm"] - invariant_rows[0]["norm"]) for row in invariant_rows
        )
        payload = {
            "schema_version": SCHEMA,
            "benchmark": "statevector_strong_scaling",
            "artifact_class": "measured_development_run",
            "benchmark_evidence_class": "non_release_smoke",
            "claim_evidence_type": "accelerator_development_performance",
            "non_release_evidence": True,
            "scalability_claim_allowed": False,
            "release_gate_allowed": False,
            "scalability_blockers": [
                "hardware_certification_development_run_not_release_evidence"
            ],
            "backend": "nccl",
            "distribution_semantics": (
                "single_device_fast_path" if world_size == 1 else "sharded_across_ranks"
            ),
            "world_size": world_size,
            "local_world_size": local_world_size,
            "node_count": node_count,
            "rank_placement": [
                {
                    "rank": row["rank"],
                    "node_rank": row["node_rank"],
                    "local_rank": row["local_rank"],
                    "hostname": row["hostname"],
                    "device": row["device"],
                }
                for row in rows
            ],
            "workload": {
                "name": "issue044_fixed_statevector_forward_v1",
                "n_wires": args.n_wires,
                "depth": args.depth,
                "gate_count": len(circuit.to_ir().instructions),
                "dtype": "complex64",
                "exchange_buffer_bytes": args.exchange_buffer_bytes,
            },
            "workload_sha256": workload_sha256,
            "timing": _timing(host),
            "device_timing": _timing(device_times),
            "rank_timings": rows,
            "rank_peak_memory_bytes": [row["peak_memory_bytes"] for row in rows],
            "communication_fraction": communication_fraction,
            "communication_bytes_per_rank_max": communication_bytes,
            "communication_tiers": {
                "route_classification": (
                    "none"
                    if not communication_bytes
                    else "inter_node_collective"
                    if node_count > 1
                    else "intra_node_collective"
                ),
                "intra_node_logical_bytes_per_rank_max": (
                    communication_bytes if node_count == 1 else 0
                ),
                "inter_node_logical_bytes_per_rank_max": (
                    communication_bytes if node_count > 1 else 0
                ),
                "physical_route": "topology_dependent_nccl_collective",
            },
            "correctness": {
                "passed": norm_error <= args.norm_tolerance
                and rank_invariant_spread <= args.norm_tolerance,
                "norm_absolute_error": norm_error,
                "rank_invariant_spread": rank_invariant_spread,
                "absolute_tolerance": args.norm_tolerance,
                "global_invariants": invariant_rows[0],
            },
            "hardware_inventory": {
                "cuda_device_names": [row["device_name"] for row in rows],
                "torch": torch.__version__,
                "python": platform.python_version(),
            },
            "gpu_activity": "cuda_events_with_device_synchronization",
            "validation_blockers": [],
        }
        payload["validation_blockers"] = list(validate_payload(payload))
        if args.json_output is not None:
            args.json_output.parent.mkdir(parents=True, exist_ok=True)
            args.json_output.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        print(json.dumps(payload, indent=2, sort_keys=True))
        if payload["validation_blockers"]:
            raise SystemExit(2)
        return payload
    finally:
        dist.destroy_process_group()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-wires", type=int, default=24)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--exchange-buffer-bytes", type=int, default=512 * 1024 * 1024)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--norm-tolerance", type=float, default=3e-5)
    parser.add_argument("--json-output", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
