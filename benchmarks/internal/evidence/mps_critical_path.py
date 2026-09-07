"""Collect the immutable ISSUE-097 site-sharded critical-path baseline."""

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
import flagquantum.experimental.distributed as fqxd
from flagquantum.runtime.backends.mps.profiling import build_mps_critical_path_report


REPO_ROOT = Path(__file__).resolve().parents[3]
MANIFEST = REPO_ROOT / "benchmarks/manifests/mps_critical_path_v1.json"


def _source_snapshot() -> dict[str, object]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False,
    ).stdout.strip()
    diff = subprocess.run(
        ["git", "diff", "--binary", "HEAD"], cwd=REPO_ROOT,
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False,
    ).stdout
    status = subprocess.run(
        ["git", "status", "--porcelain=v1"], cwd=REPO_ROOT, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False,
    ).stdout
    critical_files = (
        "flagquantum/runtime/backends/mps/profiling.py",
        "flagquantum/runtime/backends/mps/training.py",
        "flagquantum/runtime/backends/mps/reverse.py",
        "flagquantum/simulation/mps/models.py",
        "benchmarks/internal/evidence/mps_critical_path.py",
        "benchmarks/manifests/mps_critical_path_v1.json",
    )
    return {
        "source_commit": commit or "unavailable",
        "source_tree_dirty": bool(status.strip()),
        "source_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "source_status_sha256": hashlib.sha256(status.encode()).hexdigest(),
        "critical_file_sha256": {
            name: hashlib.sha256((REPO_ROOT / name).read_bytes()).hexdigest()
            for name in critical_files
        },
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
    n_wires = max(world, int(case["sites_per_rank"]) * world)
    theta = torch.tensor(0.03, device=device, requires_grad=True)
    workload = {
        "manifest_schema": manifest["schema"], "case": args.case,
        "world_size": world, "n_wires": n_wires, **case,
        "steps": manifest["warmup_steps"] + manifest["retained_warm_steps"],
        "dtype": manifest["dtype"], "optimizer": manifest["optimizer"],
    }
    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    trace_path = args.output.with_suffix(f".rank-{rank}.trace.json")
    with profile(activities=activities, record_shapes=True) as profiler:
        result = fqxd.train_distributed_mps(
            _circuit(n_wires, theta, world), steps=workload["steps"],
            observable={n_wires - 1: "z"}, optimizer="adam", lr=0.001,
            device=device, max_bond=int(case["max_bond"]),
            gradient_policy="approximate", gradient_tolerance=1e6,
            initial_mps_tensors=_initial_tensors(n_wires, int(case["initial_bond"]), device),
            initial_mps_left_canonical=True,
        )
    profiler.export_chrome_trace(str(trace_path))
    local = result.summary()
    local["trace_path"] = str(trace_path)
    local["process_group_setup_seconds"] = process_group_setup_seconds
    local["placement"] = {
        "rank": rank,
        "local_rank": local_rank,
        "hostname": socket.gethostname(),
        "device": str(device),
    }
    records = [None] * world
    dist.all_gather_object(records, local)
    if rank == 0:
        environment = {
            **_source_snapshot(), "python": platform.python_version(),
            "pytorch": torch.__version__, "backend": backend,
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
        report["trace_paths"] = [record["trace_path"] for record in records]
        report["rank_runtime_summaries"] = records
        report["manifest_path"] = str(MANIFEST.relative_to(REPO_ROOT))
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"output": str(args.output), "case": args.case, "world_size": world}))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
