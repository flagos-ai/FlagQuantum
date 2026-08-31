"""Two-node 16×A800 MPS capacity gate with 15 sharded boundaries."""

# ruff: noqa: E402, I001 -- repository path must precede an editable install.

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import flagquantum as fq  # noqa: E402
import flagquantum.experimental.distributed as fqxd  # noqa: E402
import flagquantum.experimental.mps as fqxm  # noqa: E402

from flagquantum.testing import require_general_mps_capacity  # noqa: E402

N_SITES = 24_576
MAX_BOND = 768
TARGET_WORLD = 16
TRUNCATION_BUDGET = 0.1


def bond_dimensions(n_sites: int, max_bond: int) -> tuple[int, ...]:
    return tuple(
        min(max_bond, 1 << min(index, n_sites - index))
        for index in range(n_sites + 1)
    )


def logical_mps_bytes(n_sites: int, max_bond: int) -> int:
    bonds = bond_dimensions(n_sites, max_bond)
    return sum(
        bonds[wire] * 2 * bonds[wire + 1] * 8 for wire in range(n_sites)
    )


def rank_owned_initial_mps(
    n_sites: int, max_bond: int, device: torch.device
) -> dict[int, torch.Tensor]:
    rank, world = dist.get_rank(), dist.get_world_size()
    first, last = rank * n_sites // world, (rank + 1) * n_sites // world
    bonds = bond_dimensions(n_sites, max_bond)
    generator = torch.Generator(device=device).manual_seed(520_052)
    cache: dict[tuple[int, int], torch.Tensor] = {}
    tensors = {}
    for wire in range(first, last):
        shape = (bonds[wire], bonds[wire + 1])
        if shape not in cache:
            real = torch.randn(
                2 * shape[0], shape[1], device=device, generator=generator
            )
            imag = torch.randn(
                2 * shape[0], shape[1], device=device, generator=generator
            )
            q, _ = torch.linalg.qr(torch.complex(real, imag), mode="reduced")
            cache[shape] = q.reshape(1, shape[0], 2, shape[1]).contiguous()
        tensors[wire] = cache[shape]
    return tensors


def reverse_checkpoint_capacity_bytes(logical_bytes: int, world_size: int) -> int:
    return 2 * (logical_bytes // world_size) + (1 << 30)


def cleanup_within_budget(
    allocated_bytes: int, baseline_bytes: int, peak_bytes: int
) -> tuple[bool, int]:
    runtime_cache_floor = 32 << 20
    limit = baseline_bytes + max(
        runtime_cache_floor, min(64 << 20, peak_bytes // 1000)
    )
    return allocated_bytes <= limit, limit


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def target_boundaries() -> tuple[int, ...]:
    return tuple(
        (rank + 1) * N_SITES // TARGET_WORLD - 1
        for rank in range(TARGET_WORLD - 1)
    )


def topology_fingerprint() -> str:
    content = {
        "n_sites": N_SITES,
        "bonds": bond_dimensions(N_SITES, MAX_BOND),
        "target_world_size": TARGET_WORLD,
        "target_boundaries": target_boundaries(),
    }
    return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()


def workload(theta: torch.Tensor, phi: torch.Tensor) -> fq.Circuit:
    circuit = fq.Circuit(N_SITES, device=theta.device)
    for rank in range(TARGET_WORLD):
        wire = (2 * rank + 1) * N_SITES // (2 * TARGET_WORLD)
        circuit.ry(wire, theta if rank % 2 == 0 else phi)
    for index, left in enumerate(target_boundaries()):
        circuit.rxx(left, left + 1, phi if index % 2 == 0 else theta)
    return circuit


def validate_single_gpu_baseline(payload: dict, logical_bytes: int) -> None:
    if (
        payload.get("schema") != "flagquantum.issue092.mps_capacity_baseline.v2"
        or payload.get("single_gpu_capacity_failure") is not True
        or int(payload.get("world_size", 0)) != 1
        or int(payload.get("logical_mps_bytes", 0)) != logical_bytes
        or payload.get("topology_fingerprint") != topology_fingerprint()
    ):
        raise ValueError("single-GPU baseline is unmatched or not a measured OOM")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--single-gpu-artifact", type=Path)
    parser.add_argument("--raw-log", type=Path)
    parser.add_argument("--gpu-samples", type=Path)
    args = parser.parse_args()
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world not in {1, TARGET_WORLD}:
        raise SystemExit("16-rank capacity gate supports only 1 or 16 ranks")
    if world == TARGET_WORLD and (
        args.single_gpu_artifact is None
        or args.raw_log is None
        or args.gpu_samples is None
    ):
        raise SystemExit("16-rank completion requires baseline, log, and telemetry")

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl")
    # Materialize process-lifetime NCCL CUDA state before measuring the cleanup
    # baseline. Otherwise a lower workload peak can make the leak tolerance
    # smaller than NCCL's lazy allocator footprint and reject an improvement.
    nccl_warmup = torch.zeros(1, device=device)
    dist.all_reduce(nccl_warmup)
    torch.cuda.synchronize(device)
    del nccl_warmup
    torch.cuda.empty_cache()
    baseline_allocated = int(torch.cuda.memory_allocated(device))
    torch.cuda.reset_peak_memory_stats(device)
    logical_bytes = logical_mps_bytes(N_SITES, MAX_BOND)
    status, error, result = "passed", None, None
    started = time.perf_counter()
    try:
        initial = rank_owned_initial_mps(N_SITES, MAX_BOND, device)
        theta = torch.tensor(7e-5, device=device, requires_grad=True)
        phi = torch.tensor(-1.1e-4, device=device, requires_grad=True)
        result = fqxd.train_distributed_mps(
            workload(theta, phi),
            steps=1,
            observable={N_SITES // 2: "z"},
            optimizer="adam",
            lr=0.01,
            device=device,
            max_bond=MAX_BOND,
            gradient_policy="approximate",
            gradient_tolerance=TRUNCATION_BUDGET,
            initial_mps_tensors=initial,
            initial_mps_left_canonical=True,
            canonicalization_policy="none",
            compile_site_kernels=True,
            svd_driver="gesvda",
            reverse_checkpoint_policy=fqxm.MPSReverseCheckpointPolicy(
                max_saved_bytes=reverse_checkpoint_capacity_bytes(
                    logical_bytes, world
                )
            ),
        )
        torch.cuda.synchronize(device)
    except torch.OutOfMemoryError as exc:
        status, error = "cuda_oom", str(exc).splitlines()[0]
    except RuntimeError as exc:
        status, error = "runtime_error", str(exc).splitlines()[0]

    summary = None if result is None else result.summary()
    step = None if summary is None else summary["step_metrics"][-1]
    if result is not None:
        del result
    if "initial" in locals():
        del initial
    if "theta" in locals():
        del theta
    if "phi" in locals():
        del phi
    gc.collect()
    torch.cuda.empty_cache()
    cleanup_allocated = int(torch.cuda.memory_allocated(device))
    peak_memory = int(torch.cuda.max_memory_allocated(device))
    cleanup_verified, cleanup_limit = cleanup_within_budget(
        cleanup_allocated, baseline_allocated, peak_memory
    )
    local = {
        "rank": rank,
        "status": status,
        "error": error,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_memory_bytes": peak_memory,
        "device_total_memory_bytes": int(
            torch.cuda.get_device_properties(device).total_memory
        ),
        "owned_wires": [rank * N_SITES // world, (rank + 1) * N_SITES // world],
        "useful_work": False if summary is None else summary["rank_useful_work"],
        "step": step,
        "boundary_bytes": 0 if step is None else step["boundary_bytes"],
        "two_site_splits": 0 if step is None else step["two_site_splits"],
        "cleanup_verified": cleanup_verified,
        "cleanup_allocated_bytes": cleanup_allocated,
        "cleanup_baseline_bytes": baseline_allocated,
        "cleanup_limit_bytes": cleanup_limit,
    }
    records = [None] * world
    dist.all_gather_object(records, local)

    if rank == 0 and world == 1:
        payload = {
            "schema": "flagquantum.issue092.mps_capacity_baseline.v2",
            "status": status,
            "world_size": 1,
            "batch_size": 1,
            "n_sites": N_SITES,
            "initial_max_bond": MAX_BOND,
            "trained_max_bond": MAX_BOND,
            "logical_mps_bytes": logical_bytes,
            "device_total_memory_bytes": local["device_total_memory_bytes"],
            "topology_fingerprint": topology_fingerprint(),
            "capacity_failure": status == "cuda_oom",
            "single_gpu_capacity_failure": status == "cuda_oom",
            "rank_records": records,
            "rank_record": local,
            "scalability_claim_allowed": False,
            "release_gate_allowed": False,
        }
    elif rank == 0:
        baseline = json.loads(args.single_gpu_artifact.read_text())
        validate_single_gpu_baseline(baseline, logical_bytes)
        failed = [record for record in records if record["status"] != "passed"]
        if failed:
            payload = {
                "schema": "flagquantum.issue092.general_mps_capacity_failure.v1",
                "world_size": world,
                "n_sites": N_SITES,
                "logical_mps_bytes": logical_bytes,
                "topology_fingerprint": topology_fingerprint(),
                "rank_records": records,
                "sharded_completion": False,
                "scalability_claim_allowed": False,
                "release_gate_allowed": False,
            }
        else:
            updates = {
                update["operation_id"]: update
                for record in records
                for update in record["step"]["bond_updates"]
            }.values()
            updates = tuple(updates)
            boundaries = [
                {
                    "bond": bond,
                    "ranks": [index, index + 1],
                    "forward": any(
                        update["bond"] == bond
                        and tuple(update["owner_ranks"]) == (index, index + 1)
                        and update["forward_transport"] == "batched_isend_irecv"
                        for update in updates
                    ),
                    "reverse": any(
                        update["bond"] == bond
                        and tuple(update["owner_ranks"]) == (index, index + 1)
                        and update["reverse_transport"] == "batched_isend_irecv"
                        for update in updates
                    ),
                    "bond_update": next(
                        update
                        for update in updates
                        if update["bond"] == bond
                        and tuple(update["owner_ranks"]) == (index, index + 1)
                    ),
                }
                for index, bond in enumerate(target_boundaries())
            ]
            payload = {
                "schema": "flagquantum.issue092.general_mps_capacity.v1",
                "batch_size": 1,
                "n_sites": N_SITES,
                "initial_max_bond": MAX_BOND,
                "trained_max_bond": MAX_BOND,
                "logical_mps_bytes": logical_bytes,
                "topology_fingerprint": topology_fingerprint(),
                "world_size": TARGET_WORLD,
                "distribution_semantics": "sharded_across_ranks",
                "single_gpu_capacity_failure": True,
                "sharded_completion": True,
                "full_mps_materialization": False,
                "boundary_evidence": boundaries,
                "bond_dimension_changed": True,
                "gradient_policy": "approximate",
                "svd_driver": "gesvda",
                "discarded_weight": sum(
                    update["discarded_weight"] for update in updates
                ),
                "truncation_error_budget": TRUNCATION_BUDGET,
                "rank_records": records,
                "single_gpu_peak_memory_bytes": baseline["rank_record"][
                    "peak_memory_bytes"
                ],
                "single_gpu_device_total_memory_bytes": baseline[
                    "device_total_memory_bytes"
                ],
                "scalability_claim_allowed": False,
                "release_gate_allowed": False,
                "command": " ".join(sys.argv),
                "commit": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
                ).strip(),
                "source_artifacts": [
                    {
                        "kind": "single_gpu_failure",
                        "path": str(args.single_gpu_artifact),
                        "sha256": sha256(args.single_gpu_artifact),
                    },
                    {
                        "kind": "workload",
                        "path": "benchmarks/internal/evidence/general_mps_capacity_16.py",
                        "sha256": sha256(Path(__file__).resolve()),
                    },
                    {
                        "kind": "raw_log",
                        "path": str(args.raw_log),
                        "sha256": sha256(args.raw_log),
                    },
                    {
                        "kind": "gpu_samples",
                        "path": str(args.gpu_samples),
                        "sha256": sha256(args.gpu_samples),
                    },
                ],
            }
            require_general_mps_capacity(payload)
    if rank == 0:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"output": str(args.output), "status": status}), flush=True)
    dist.destroy_process_group()
    if status != "passed" and world == TARGET_WORLD:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
