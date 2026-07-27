"""Train one rank-sharded, variable-bond MPS that exceeds one A100 40GB.

The capacity profile represents a 12,288-site translationally invariant random
isometric MPS with maximum bond dimension 512.  It applies a shared trainable
transverse-field rotation and evaluates a center-site Z objective.  This is one
logical quantum state (batch size one), not data-parallel replicas.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import threading
import time
from pathlib import Path

import torch
import torch.distributed as dist

import flagquantum as fq


PROFILES = {
    "smoke": (32, 8),
    "capacity": (12_288, 512),
}


def bond_dimensions(n_sites: int, max_bond: int) -> tuple[int, ...]:
    """Open-boundary dimensions that grow/decay by at most physical dimension 2."""
    return tuple(
        min(max_bond, 1 << min(index, n_sites - index))
        for index in range(n_sites + 1)
    )


def logical_mps_bytes(n_sites: int, max_bond: int) -> int:
    bonds = bond_dimensions(n_sites, max_bond)
    return sum(bonds[wire] * 2 * bonds[wire + 1] * 8 for wire in range(n_sites))


def _isometry(
    left: int,
    right: int,
    *,
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:
    real = torch.randn(2 * left, right, device=device, generator=generator)
    imag = torch.randn(2 * left, right, device=device, generator=generator)
    matrix = torch.complex(real, imag)
    q, _ = torch.linalg.qr(matrix, mode="reduced")
    return q.reshape(1, left, 2, right).contiguous()


def rank_owned_initial_mps(
    n_sites: int, max_bond: int, device: torch.device
) -> dict[int, torch.Tensor]:
    rank, world = dist.get_rank(), dist.get_world_size()
    first = rank * n_sites // world
    last = (rank + 1) * n_sites // world
    bonds = bond_dimensions(n_sites, max_bond)
    generator = torch.Generator(device=device).manual_seed(520_052)
    cache: dict[tuple[int, int], torch.Tensor] = {}
    tensors = {}
    for wire in range(first, last):
        shape = (bonds[wire], bonds[wire + 1])
        if shape not in cache:
            cache[shape] = _isometry(
                *shape, device=device, generator=generator
            )
        # Reusing the translationally invariant site tensor is intentional.
        # The runtime clones every logical site into rank-owned storage.
        tensors[wire] = cache[shape]
    return tensors


def build_quench(n_sites: int, theta: torch.Tensor) -> fq.Circuit:
    circuit = fq.Circuit(n_sites, device=theta.device)
    for wire in range(n_sites):
        circuit.ry(wire, theta)
    return circuit


def heartbeat(stop: threading.Event, rank: int, device: torch.device) -> None:
    started = time.monotonic()
    while not stop.wait(30):
        if rank == 0:
            print(
                json.dumps(
                    {
                        "phase": "distributed_mps_training",
                        "elapsed_seconds": round(time.monotonic() - started, 1),
                        "rank0_allocated_bytes": torch.cuda.memory_allocated(device),
                    }
                ),
                flush=True,
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=PROFILES, default="smoke")
    parser.add_argument("--n-sites", type=int)
    parser.add_argument("--max-bond", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    default_sites, default_bond = PROFILES[args.profile]
    n_sites = args.n_sites or default_sites
    max_bond = args.max_bond or default_bond
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if args.profile == "capacity" and world not in {1, 8}:
        raise SystemExit("capacity profile requires world size 1 (OOM baseline) or 8")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl")
    stop = threading.Event()
    monitor = threading.Thread(target=heartbeat, args=(stop, rank, device), daemon=True)
    monitor.start()
    status = "passed"
    error = None
    result = None
    started = time.perf_counter()
    logical_bytes = logical_mps_bytes(n_sites, max_bond)
    try:
        initial = rank_owned_initial_mps(n_sites, max_bond, device)
        theta = torch.tensor(0.07, device=device, requires_grad=True)
        result = fq.train_distributed_mps(
            build_quench(n_sites, theta),
            steps=1,
            observable={n_sites // 2: "z"},
            optimizer="adam",
            lr=0.01,
            device=device,
            initial_mps_tensors=initial,
            initial_mps_left_canonical=True,
            reverse_checkpoint_policy=fq.MPSReverseCheckpointPolicy(
                max_saved_bytes=max(256 << 20, 2 * logical_bytes // world)
            ),
            memory_leak_tolerance_bytes=64 << 20,
        )
        torch.cuda.synchronize(device)
    except torch.OutOfMemoryError as exc:
        status = "cuda_oom"
        error = str(exc).splitlines()[0]
    finally:
        stop.set()
        monitor.join(timeout=2)
    if status == "cuda_oom":
        result = None
        if "initial" in locals():
            del initial
        gc.collect()
        torch.cuda.empty_cache()
    local = {
        "rank": rank,
        "status": status,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        "owned_wires": [rank * n_sites // world, (rank + 1) * n_sites // world],
        "error": error,
        "training": None if result is None else result.summary(),
    }
    if world == 1:
        records = [local]
    else:
        records = [None] * world
        dist.all_gather_object(records, local)
    if rank == 0:
        payload = {
            "schema": "flagquantum.example.variable_bond_capacity_8gpu.v1",
            "workload": "single_state_transverse_field_quench",
            "batch_size": 1,
            "n_sites": n_sites,
            "max_bond": max_bond,
            "logical_mps_bytes": logical_bytes,
            "world_size": world,
            "distribution_semantics": (
                "single_device_fast_path" if world == 1 else "sharded_across_ranks"
            ),
            "full_mps_materialization": False,
            "rank_records": records,
            "single_gpu_capacity_failure": world == 1 and status == "cuda_oom",
            "sharded_completion": world == 8 and all(
                record["status"] == "passed" for record in records
            ),
            "scalability_claim_allowed": False,
            "release_gate_allowed": False,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"output": str(args.output), "status": status}), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
