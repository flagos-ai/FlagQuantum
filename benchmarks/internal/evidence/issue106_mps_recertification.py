"""Rerun the frozen ISSUE-097 matrix with ISSUE-098..105 optimizations enabled."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import socket
import subprocess
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.profiler import ProfilerActivity, profile

import flagquantum as fq
from flagquantum.runtime.backends.mps.profiling import build_mps_critical_path_report
from flagquantum.runtime.backends.mps.site_kernels import (
    reset_site_kernel_stats,
    site_kernel_cache_events,
    site_kernel_stats,
)


ROOT = Path(__file__).resolve().parents[3]
MANIFEST = ROOT / "benchmarks/manifests/issue097_mps_critical_path_v1.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _snapshot() -> dict[str, object]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain=v1"], cwd=ROOT, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False,
    ).stdout
    files = (
        "flagquantum/runtime/backends/mps/training.py",
        "flagquantum/runtime/backends/mps/reverse.py",
        "flagquantum/runtime/backends/mps/site_kernels.py",
        "flagquantum/runtime/backends/mps/profiling.py",
        "benchmarks/internal/evidence/issue106_mps_recertification.py",
        "benchmarks/manifests/issue097_mps_critical_path_v1.json",
    )
    return {
        "source_commit": commit or "unavailable",
        "source_tree_dirty": bool(status.strip()),
        "source_status_sha256": hashlib.sha256(status.encode()).hexdigest(),
        "critical_file_sha256": {name: _sha(ROOT / name) for name in files},
    }


def _initial_tensors(n_wires: int, bond: int, device: torch.device):
    rank, world = dist.get_rank(), dist.get_world_size()
    base, extra = divmod(n_wires, world)
    starts = [item * base + min(item, extra) for item in range(world)]
    stops = [start + base + (item < extra) for item, start in enumerate(starts)]
    tensors = {}
    generator = torch.Generator(device=device).manual_seed(9700 + rank)
    for wire in range(starts[rank], stops[rank]):
        left = 1 if wire == 0 else bond
        right = 1 if wire == n_wires - 1 else bond
        real = torch.randn(1, left, 2, right, generator=generator, device=device)
        imag = torch.randn(1, left, 2, right, generator=generator, device=device)
        tensors[wire] = torch.complex(real, imag) / max(1, bond)
    return tensors


def _circuit(n_wires: int, theta: torch.Tensor, world: int):
    circuit = fq.Circuit(n_wires, device=theta.device)
    for wire in range(n_wires):
        circuit.ry(wire, theta)
    for rank in range(world - 1):
        boundary = (rank + 1) * n_wires // world - 1
        circuit.rxx(boundary, boundary + 1, theta)
    return circuit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=("small_latency", "crossover", "large_bond"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_text())
    case = manifest["cases"][args.case]
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device("cuda", local_rank) if backend == "nccl" else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    process_group_started = time.perf_counter()
    dist.init_process_group(backend, device_id=device if backend == "nccl" else None)
    process_group_setup_seconds = time.perf_counter() - process_group_started
    rank, world = dist.get_rank(), dist.get_world_size()
    compile_site_kernels = world > 1
    n_wires = max(world, int(case["sites_per_rank"]) * world)
    theta = torch.tensor(0.03, device=device, requires_grad=True)
    workload = {
        "manifest_schema": manifest["schema"], "case": args.case,
        "world_size": world, "n_wires": n_wires, **case,
        "steps": manifest["warmup_steps"] + manifest["retained_warm_steps"],
        "dtype": manifest["dtype"], "optimizer": manifest["optimizer"],
        "optimization_set": "issue098_through_issue105",
        "site_kernel_policy": "compiled_distributed" if compile_site_kernels else "eager_local_fast_path",
    }
    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    trace_path = args.output.with_suffix(f".rank-{rank}.trace.json")
    reset_site_kernel_stats(clear_cache=True)
    with profile(activities=activities, record_shapes=True) as profiler:
        result = fq.train_distributed_mps(
            _circuit(n_wires, theta, world), steps=workload["steps"],
            observable={n_wires - 1: "z"}, optimizer="adam", lr=0.001,
            device=device, max_bond=int(case["max_bond"]),
            gradient_policy="approximate", gradient_tolerance=1e6,
            initial_mps_tensors=_initial_tensors(n_wires, int(case["initial_bond"]), device),
            initial_mps_left_canonical=True,
            compile_site_kernels=compile_site_kernels,
            compile_observables=compile_site_kernels,
            fuse_local_reverse=True, canonicalization_policy="dirty",
        )
    profiler.export_chrome_trace(str(trace_path))
    averages = tuple(profiler.key_averages())
    cuda_activity_us = sum(float(getattr(item, "self_device_time_total", 0.0)) for item in averages)
    communication_us = sum(
        float(getattr(item, "self_device_time_total", 0.0))
        for item in averages
        if item.key in {
            "flagquantum::mps::p2p_send", "flagquantum::mps::p2p_recv",
            "flagquantum::mps::gradient_all_reduce",
        }
    )
    local = result.summary()
    local.update(
        {
            "trace_path": str(trace_path),
            "site_kernel_stats": site_kernel_stats(),
            "site_kernel_event_counts": {
                name: sum(event["event"] == name for event in site_kernel_cache_events())
                for name in ("hit", "miss", "compile", "eviction", "unsupported_shape_fallback")
            },
            "profiled_cuda_activity_seconds": cuda_activity_us / 1e6,
            "profiled_communication_seconds": communication_us / 1e6,
            "losses_finite": all(torch.isfinite(torch.tensor(value)) for value in result.losses),
            "process_group_setup_seconds": process_group_setup_seconds,
            "placement": {
                "rank": rank,
                "local_rank": local_rank,
                "hostname": socket.gethostname(),
                "device": str(device),
            },
        }
    )
    records = [None] * world
    dist.all_gather_object(records, local)
    if rank == 0:
        environment = {
            **_snapshot(), "python": platform.python_version(), "pytorch": torch.__version__,
            "backend": backend,
            "device": torch.cuda.get_device_name(0) if device.type == "cuda" else platform.processor(),
            "cuda": torch.version.cuda, "world_size": world,
            "local_world_size": int(os.environ.get("LOCAL_WORLD_SIZE", str(world))),
            "node_count": len({record["placement"]["hostname"] for record in records}),
            "rank_placement": [record["placement"] for record in records],
        }
        report = build_mps_critical_path_report(
            records, workload_manifest=workload, environment_manifest=environment,
            warmup_steps=int(manifest["warmup_steps"]),
        )
        warm_end_to_end = sum(
            sample["end_to_end_seconds"] for sample in report["rank_step_samples"]
            if sample["sample_class"] == "warm"
        )
        report.update(
            {
                "schema": "flagquantum.issue106.mps_recertification_run.v1",
                "trace_paths": [record["trace_path"] for record in records],
                "manifest_path": str(MANIFEST.relative_to(ROOT)),
                "manifest_sha256": _sha(MANIFEST),
                "rank_runtime_summaries": records,
                "compile_setup_seconds_max": max(record["site_kernel_stats"]["compile_seconds"] for record in records),
                "compile_amortization_fraction": (
                    sum(record["site_kernel_stats"]["compile_seconds"] for record in records)
                    / max(warm_end_to_end, 1e-12)
                ),
                "all_losses_finite": all(record["losses_finite"] for record in records),
                "correctness_reference": "benchmarks/development/distributed_mps_correctness_matrix.json",
                "capacity_reference": "benchmarks/development/issue092_general_mps_capacity.json",
                "recovery_reference": "benchmarks/development/issue093_mps_stability_matrix.json",
                "performance_claim_allowed": False,
                "scalability_claim_allowed": False,
            }
        )
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"output": str(args.output), "case": args.case, "world_size": world}))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
