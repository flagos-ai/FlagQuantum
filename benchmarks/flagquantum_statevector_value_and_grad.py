#!/usr/bin/env python3
"""Matched FlagQuantum distributed value-and-full-gradient benchmark."""

from __future__ import annotations

import argparse
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

from benchmarks.gpu_memory_sampler import NvmlMemorySampler  # noqa: E402
from benchmarks.statevector_training_scaling import (  # noqa: E402
    build_full_width_workload,
)
from flagquantum.runtime.backends.statevector.reverse import (  # noqa: E402
    StatevectorCheckpointPolicy,
    execute_torch_distributed_statevector_reverse,
)

SCHEMA = "flagquantum.statevector.value_and_full_gradient.v1"


def _timing(samples: list[float]) -> dict[str, Any]:
    mean = statistics.fmean(samples)
    return {
        "samples_seconds": samples,
        "sample_count": len(samples),
        "median_seconds": statistics.median(samples),
        "mean_seconds": mean,
        "coefficient_of_variation": statistics.pstdev(samples) / mean,
    }


def _rank_max(value: float, device: torch.device) -> float:
    packed = torch.tensor(value, dtype=torch.float64, device=device)
    dist.all_reduce(packed, op=dist.ReduceOp.MAX)
    return float(packed.cpu())


def _profile_metrics(
    profile: torch.profiler.profile,
    *,
    wall_seconds: float,
    runtime_summary: dict[str, Any],
) -> dict[str, Any]:
    events = list(profile.events())
    cuda_events = [
        event
        for event in events
        if event.device_type == torch.autograd.DeviceType.CUDA
    ]
    communication_events = [
        event
        for event in cuda_events
        if "nccl" in event.name.lower() or "c10d" in event.name.lower()
    ]
    intervals = sorted(
        (
            float(event.time_range.start),
            float(event.time_range.end),
        )
        for event in cuda_events
        if event.time_range.end > event.time_range.start
    )
    busy_us = 0.0
    if intervals:
        start, end = intervals[0]
        for next_start, next_end in intervals[1:]:
            if next_start <= end:
                end = max(end, next_end)
            else:
                busy_us += end - start
                start, end = next_start, next_end
        busy_us += end - start
    communication_us = sum(
        float(event.time_range.end - event.time_range.start)
        for event in communication_events
    )
    communication_bytes = int(
        runtime_summary.get("backward_communication_bytes", 0)
    )
    return {
        "measurement": "one additional post-protocol torch.profiler step",
        "kernel_launch_count": len(cuda_events),
        "gpu_busy_time_seconds_union": busy_us / 1e6,
        "profile_wall_seconds": wall_seconds,
        "gpu_activity_fraction": min(1.0, busy_us / 1e6 / wall_seconds),
        "communication_kernel_count": len(communication_events),
        "communication_device_time_seconds_sum": communication_us / 1e6,
        "logical_communication_bytes": communication_bytes,
        "effective_logical_bandwidth_bytes_per_second": (
            communication_bytes / (communication_us / 1e6)
            if communication_us > 0
            else None
        ),
        "communication_compute_overlap_observed": bool(
            communication_us > 0 and busy_us < sum(
                float(event.time_range.end - event.time_range.start)
                for event in cuda_events
            )
        ),
    }


def run(args: argparse.Namespace) -> dict[str, Any] | None:
    if args.n_wires < 2 or args.layers < 1:
        raise ValueError("n_wires >= 2 and layers >= 1 required")
    if args.warmup < 0 or args.repetitions < 3:
        raise ValueError("warmup >= 0 and repetitions >= 3 required")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    rank, world = dist.get_rank(), dist.get_world_size()
    os.environ["FQ_STATEVECTOR_COMM_AWARE_LAYOUT"] = "1"
    os.environ["FQ_STATEVECTOR_REVERSE_EXCHANGE_WORKSPACE"] = "1"
    os.environ["FQ_STATEVECTOR_TRITON_VJP_ADJOINT"] = "1"
    os.environ["FQ_STATEVECTOR_FUSED_VJP_PIPELINE"] = "1"
    os.environ["FQ_STATEVECTOR_CROSS_SHARD_CX_PACK"] = "1"
    os.environ["FQ_STATEVECTOR_TOPOLOGY_AWARE_RANK_BITS"] = "1"
    try:
        circuit, parameters, metadata = build_full_width_workload(
            args.n_wires,
            device,
            layers=args.layers,
            seed=args.seed,
            entanglement=args.entanglement,
        )

        def value_and_grad() -> tuple[
            float, tuple[float, ...], float, float, int, int, dict
        ]:
            for parameter in parameters:
                parameter.grad = None
            dist.barrier()
            with NvmlMemorySampler(local_rank) as memory:
                torch.cuda.synchronize(device)
                forward_started = time.perf_counter()
                result = execute_torch_distributed_statevector_reverse(
                    circuit,
                    observable_wire=args.n_wires // 2,
                    checkpoint_policy=StatevectorCheckpointPolicy(
                        strategy="reversible_adjoint"
                    ),
                    device=device,
                )
                torch.cuda.synchronize(device)
                forward_seconds = _rank_max(
                    time.perf_counter() - forward_started, device
                )
                backward_started = time.perf_counter()
                result.backward()
                torch.cuda.synchronize(device)
                backward_seconds = _rank_max(
                    time.perf_counter() - backward_started, device
                )
            peak = torch.tensor(memory.peak_bytes, dtype=torch.int64, device=device)
            peak_sum = peak.clone()
            dist.all_reduce(peak, op=dist.ReduceOp.MAX)
            dist.all_reduce(peak_sum, op=dist.ReduceOp.SUM)
            gradients = tuple(float(parameter.grad.detach()) for parameter in parameters)
            return (
                float(result.value.detach()),
                gradients,
                forward_seconds,
                backward_seconds,
                int(peak.cpu()),
                int(peak_sum.cpu()),
                result.summary(),
            )

        initial_value = None
        initial_gradients = None
        summary = None
        for index in range(args.warmup):
            initial_value, initial_gradients, _, _, _, _, summary = value_and_grad()
            if rank == 0:
                print(
                    f"heartbeat phase=warmup step={index + 1}/{args.warmup} "
                    f"entanglement={args.entanglement}",
                    flush=True,
                )
        forward_samples: list[float] = []
        backward_samples: list[float] = []
        total_samples: list[float] = []
        peak_memory_per_rank_samples: list[int] = []
        peak_memory_aggregate_samples: list[int] = []
        for index in range(args.repetitions):
            (
                value,
                gradients,
                forward,
                backward,
                peak_per_rank,
                peak_aggregate,
                summary,
            ) = value_and_grad()
            if initial_value is None:
                initial_value, initial_gradients = value, gradients
            forward_samples.append(forward)
            backward_samples.append(backward)
            total_samples.append(forward + backward)
            peak_memory_per_rank_samples.append(peak_per_rank)
            peak_memory_aggregate_samples.append(peak_aggregate)
            if rank == 0:
                print(
                    f"heartbeat phase=measurement step={index + 1}/"
                    f"{args.repetitions} forward={forward:.6f}s "
                    f"backward={backward:.6f}s",
                    flush=True,
                )
        profile_by_rank = None
        if args.profile:
            dist.barrier()
            profile_started = time.perf_counter()
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ]
            ) as profile:
                *_, profile_summary = value_and_grad()
            torch.cuda.synchronize(device)
            local_profile = _profile_metrics(
                profile,
                wall_seconds=time.perf_counter() - profile_started,
                runtime_summary=profile_summary,
            )
            profile_by_rank = [None] * world
            dist.all_gather_object(profile_by_rank, local_profile)
        assert initial_value is not None and initial_gradients is not None
        assert summary is not None
        finite = bool(
            torch.isfinite(torch.tensor([initial_value, *initial_gradients])).all()
        )
        placement = {
            "rank": rank,
            "local_rank": local_rank,
            "hostname": platform.node(),
        }
        placements: list[dict[str, Any] | None] = [None] * world
        dist.all_gather_object(placements, placement)
        if rank != 0:
            return None
        hosts = {item["hostname"] for item in placements if item is not None}
        return {
            "schema": SCHEMA,
            "benchmark": "flagquantum_adjoint_value_and_full_gradient",
            "artifact_class": "measured_development_run",
            "claim_evidence_type": "value_and_gradient_comparison",
            "release_gate_allowed": False,
            "scalability_claim_allowed": False,
            "distribution_semantics": (
                "sharded_across_ranks" if world > 1 else "single_device_fast_path"
            ),
            "world_size": world,
            "local_world_size": int(os.environ.get("LOCAL_WORLD_SIZE", world)),
            "node_count": len(hosts),
            "rank_placement": placements,
            "environment": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "device_name": torch.cuda.get_device_name(local_rank),
            },
            "workload": {
                "name": metadata["name"],
                "n_wires": args.n_wires,
                "layers": args.layers,
                "gate_count": metadata["gate_count"],
                "parameter_count": metadata["parameter_count"],
                "gate_set": ["RY", "CNOT"],
                "entanglement": metadata["entanglement"],
                "observable": f"Z({args.n_wires // 2})",
                "dtype": "complex64",
                "parameter_dtype": "float32",
                "seed": args.seed,
            },
            "protocol": {
                "gradient_method": "reversible_adjoint",
                "fixed_parameters_across_samples": True,
                "optimizer_included": False,
                "warmup": args.warmup,
                "repetitions": args.repetitions,
                "synchronization": "cuda_device_synchronized_wall_clock",
                "distributed_sample_reduction": "rank_max",
                "triton_vjp_adjoint": True,
                "communication_aware_layout": True,
                "packed_gradient_all_reduce": True,
                "persistent_wire_layout": os.getenv(
                    "FQ_STATEVECTOR_PERSISTENT_WIRE_LAYOUT", "1"
                ),
                "persistent_inplace_local": os.getenv(
                    "FQ_STATEVECTOR_PERSISTENT_INPLACE_LOCAL", "1"
                ),
                "local_block_fusion": os.getenv(
                    "FQ_STATEVECTOR_LOCAL_BLOCK_FUSION", "1"
                ),
                "local_block_fusion_width": os.getenv(
                    "FQ_STATEVECTOR_LOCAL_BLOCK_FUSION_WIDTH", "3"
                ),
                "triton_local_cx": os.getenv(
                    "FQ_STATEVECTOR_TRITON_LOCAL_CX", "0"
                ),
                "triton_transpose_1q": os.getenv(
                    "FQ_STATEVECTOR_TRITON_TRANSPOSE_1Q", "1"
                ),
                "triton_cx_segment": os.getenv(
                    "FQ_STATEVECTOR_TRITON_CX_SEGMENT", "auto"
                ),
                "ket_checkpoint_mode": os.getenv(
                    "FQ_STATEVECTOR_INTER_NODE_KET_CHECKPOINTS", "inter_node"
                ),
                "nccl_min_p2p_nchannels": os.getenv(
                    "NCCL_MIN_P2P_NCHANNELS", "auto"
                ),
                "nccl_nchannels_per_net_peer": os.getenv(
                    "NCCL_NCHANNELS_PER_NET_PEER", "auto"
                ),
                "single_pass_reversible_rotation_vjp": True,
            },
            "correctness": {
                "initial_value": initial_value,
                "initial_gradients": initial_gradients,
                "values_and_gradients_finite": finite,
            },
            "forward": _timing(forward_samples),
            "backward": _timing(backward_samples),
            "value_and_grad": _timing(total_samples),
            "memory": {
                "measurement": "NVML device-used bytes sampled every 20 ms",
                "exclusive_gpu_required": True,
                "peak_bytes_per_rank_max": max(peak_memory_per_rank_samples),
                "peak_bytes_aggregate_max": max(peak_memory_aggregate_samples),
                "samples_per_rank_max_bytes": peak_memory_per_rank_samples,
                "samples_aggregate_bytes": peak_memory_aggregate_samples,
            },
            "runtime_summary": summary,
            "profiler": {
                "enabled": bool(args.profile),
                "rank_metrics": profile_by_rank,
                "timing_samples_exclude_profile_step": True,
            },
        }
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-wires", type=int, required=True)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument(
        "--entanglement",
        choices=("linear", "ring", "brickwork"),
        default="linear",
    )
    parser.add_argument("--seed", type=int, default=440044)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=30)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--json-output", type=Path, required=True)
    args = parser.parse_args()
    payload = run(args)
    if payload is None:
        return
    encoded = json.dumps(payload, indent=2, sort_keys=True)
    print(encoded)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(encoded + "\n", encoding="utf-8")
    if not payload["correctness"]["values_and_gradients_finite"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
