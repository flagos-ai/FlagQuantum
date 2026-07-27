"""Batch-one entangling variable-bond MPS capacity gate for ISSUE-092."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import torch
import torch.distributed as dist

import flagquantum as fq

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.distributed_mps.variable_bond_capacity_8gpu import (  # noqa: E402
    bond_dimensions,
    logical_mps_bytes,
    rank_owned_initial_mps,
)
from flagquantum.testing import require_general_mps_capacity  # noqa: E402
from flagquantum.testing.mps_capacity_certification import (  # noqa: E402
    ISSUE091_TRUNCATION_BUDGET,
)

PROFILES = {"smoke": (32, 8, 7), "capacity": (12_288, 512, 511)}
TARGET_WORLD = 8
TRUNCATION_BUDGET = ISSUE091_TRUNCATION_BUDGET


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def target_boundaries(n_sites: int) -> tuple[int, ...]:
    return tuple((rank + 1) * n_sites // TARGET_WORLD - 1 for rank in range(7))


def build_entangling_workload(n_sites: int, theta, phi) -> fq.Circuit:
    circuit = fq.Circuit(n_sites, device=theta.device)
    for rank in range(TARGET_WORLD):
        wire = (2 * rank + 1) * n_sites // (2 * TARGET_WORLD)
        circuit.ry(wire, theta if rank % 2 == 0 else phi)
    for index, left in enumerate(target_boundaries(n_sites)):
        if index % 2 == 0:
            circuit.rxx(left, left + 1, phi)
        else:
            circuit.rzz(left, left + 1, theta)
    return circuit


def topology_fingerprint(n_sites: int, max_bond: int) -> str:
    content = {
        "n_sites": n_sites,
        "bonds": bond_dimensions(n_sites, max_bond),
        "target_world_size": TARGET_WORLD,
        "target_boundaries": target_boundaries(n_sites),
    }
    return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()


def heartbeat(stop, rank, device):
    started = time.monotonic()
    while not stop.wait(30):
        if rank == 0:
            print(
                json.dumps(
                    {
                        "phase": "issue092_entangling_forward_reverse",
                        "elapsed_seconds": round(time.monotonic() - started, 1),
                        "rank0_allocated_bytes": torch.cuda.memory_allocated(device),
                        "rank0_reserved_bytes": torch.cuda.memory_reserved(device),
                    }
                ),
                flush=True,
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=PROFILES, default="smoke")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--single-gpu-artifact", type=Path)
    parser.add_argument("--raw-log", type=Path)
    parser.add_argument("--gpu-samples", type=Path)
    parser.add_argument("--steps", type=int, default=1)
    args = parser.parse_args()
    n_sites, initial_bond, trained_bond = PROFILES[args.profile]
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if args.profile == "capacity" and world not in {1, 8}:
        raise SystemExit("capacity profile requires one or eight ranks")
    if args.profile == "capacity" and world == 8 and args.single_gpu_artifact is None:
        raise SystemExit("eight-rank run requires --single-gpu-artifact")
    if args.profile == "capacity" and world == 8 and (
        args.raw_log is None or args.gpu_samples is None
    ):
        raise SystemExit("eight-rank run requires --raw-log and --gpu-samples")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl")
    stop = threading.Event()
    monitor = threading.Thread(target=heartbeat, args=(stop, rank, device), daemon=True)
    monitor.start()
    status, error, result = "passed", None, None
    started = time.perf_counter()
    logical_bytes = logical_mps_bytes(n_sites, initial_bond)
    try:
        initial = rank_owned_initial_mps(n_sites, initial_bond, device)
        theta = torch.tensor(0.07, device=device, requires_grad=True)
        phi = torch.tensor(-0.11, device=device, requires_grad=True)
        result = fq.train_distributed_mps(
            build_entangling_workload(n_sites, theta, phi),
            steps=args.steps,
            observable={n_sites // 2: "z"},
            optimizer="adam",
            lr=0.01,
            device=device,
            max_bond=trained_bond,
            gradient_policy="approximate",
            gradient_tolerance=TRUNCATION_BUDGET,
            initial_mps_tensors=initial,
            initial_mps_left_canonical=True,
            reverse_checkpoint_policy=fq.MPSReverseCheckpointPolicy(
                max_saved_bytes=max(256 << 20, logical_bytes // world)
            ),
        )
        torch.cuda.synchronize(device)
    except torch.OutOfMemoryError as exc:
        status, error = "cuda_oom", str(exc).splitlines()[0]
    except RuntimeError as exc:
        status, error = "runtime_error", str(exc).splitlines()[0]
    finally:
        stop.set()
        monitor.join(timeout=2)
    if status != "passed":
        result = None
        if "initial" in locals():
            del initial
        gc.collect()
        torch.cuda.empty_cache()
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
    local = {
        "rank": rank,
        "status": status,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        "owned_wires": [rank * n_sites // world, (rank + 1) * n_sites // world],
        "useful_work": False if summary is None else summary["rank_useful_work"],
        "step": step,
        "memory_timeline": (
            []
            if summary is None
            else [
                {
                    "step": item["step"],
                    "allocated": item["allocated_memory_bytes"],
                    "reserved": item["reserved_memory_bytes"],
                    "peak": item["peak_memory_bytes"],
                }
                for item in summary["step_metrics"]
            ]
        ),
        "error": error,
        "boundary_bytes": 0 if step is None else step["boundary_bytes"],
        "two_site_splits": 0 if step is None else step["two_site_splits"],
        "cleanup_verified": cleanup_allocated == 0,
        "cleanup_allocated_bytes": cleanup_allocated,
    }
    if world == 1:
        records = [local]
    else:
        records = [None] * world
        dist.all_gather_object(records, local)
    if rank == 0:
        if world == 1:
            payload = {
                "schema": "flagquantum.issue092.single_gpu_capacity_failure.v1",
                "status": status,
                "batch_size": 1,
                "n_sites": n_sites,
                "initial_max_bond": initial_bond,
                "trained_max_bond": trained_bond,
                "logical_mps_bytes": logical_bytes,
                "topology_fingerprint": topology_fingerprint(n_sites, initial_bond),
                "single_gpu_capacity_failure": status == "cuda_oom",
                "rank_record": local,
            }
        else:
            baseline = (
                json.loads(args.single_gpu_artifact.read_text())
                if args.single_gpu_artifact is not None
                else {"single_gpu_capacity_failure": False}
            )
            if (
                baseline.get("schema") != "flagquantum.issue092.single_gpu_capacity_failure.v1"
                or baseline.get("status") != "cuda_oom"
                or baseline.get("topology_fingerprint")
                != topology_fingerprint(n_sites, initial_bond)
                or int(baseline.get("logical_mps_bytes", 0)) != logical_bytes
            ):
                raise SystemExit("single-GPU artifact does not match this workload")
            failed = [record for record in records if record["status"] != "passed"]
            if failed:
                payload = {
                    "schema": "flagquantum.issue092.general_mps_capacity_failure.v1",
                    "world_size": world,
                    "n_sites": n_sites,
                    "initial_max_bond": initial_bond,
                    "trained_max_bond": trained_bond,
                    "logical_mps_bytes": logical_bytes,
                    "topology_fingerprint": topology_fingerprint(
                        n_sites, initial_bond
                    ),
                    "rank_records": records,
                    "sharded_completion": False,
                    "scalability_claim_allowed": False,
                    "release_gate_allowed": False,
                }
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(
                    json.dumps(payload, indent=2, sort_keys=True) + "\n"
                )
                print(json.dumps({"output": str(args.output), "status": "failed"}))
                dist.destroy_process_group()
                raise SystemExit(2)
            updates = [
                update
                for record in records
                for update in record["step"]["bond_updates"]
            ]
            unique_updates = tuple(
                {update["operation_id"]: update for update in updates}.values()
            )
            boundary_evidence = [
                {
                    "bond": left,
                    "ranks": [index, index + 1],
                    "forward": any(
                        update["bond"] == left
                        and tuple(update["owner_ranks"]) == (index, index + 1)
                        and update["forward_transport"] == "batched_isend_irecv"
                        for update in unique_updates
                    ),
                    "reverse": any(
                        update["bond"] == left
                        and tuple(update["owner_ranks"]) == (index, index + 1)
                        and update["reverse_transport"] == "batched_isend_irecv"
                        for update in unique_updates
                    ),
                    "bond_update": next(
                        (
                            update
                            for update in unique_updates
                            if update["bond"] == left
                            and tuple(update["owner_ranks"]) == (index, index + 1)
                        ),
                        {},
                    ),
                }
                for index, left in enumerate(target_boundaries(n_sites))
            ]
            discarded = sum(update["discarded_weight"] for update in unique_updates)
            payload = {
                "schema": (
                    "flagquantum.issue092.general_mps_capacity.v1"
                    if args.profile == "capacity"
                    else "flagquantum.issue092.general_mps_capacity_smoke.v1"
                ),
                "batch_size": 1,
                "n_sites": n_sites,
                "initial_max_bond": initial_bond,
                "trained_max_bond": trained_bond,
                "logical_mps_bytes": logical_bytes,
                "topology_fingerprint": topology_fingerprint(n_sites, initial_bond),
                "world_size": world,
                "distribution_semantics": "sharded_across_ranks",
                "single_gpu_capacity_failure": baseline["single_gpu_capacity_failure"],
                "single_gpu_peak_memory_bytes": baseline["rank_record"]["peak_memory_bytes"],
                "single_gpu_artifact": str(args.single_gpu_artifact),
                "sharded_completion": all(record["status"] == "passed" for record in records),
                "full_mps_materialization": False,
                "boundary_evidence": boundary_evidence,
                "bond_dimension_changed": any(
                    update["kept_rank"] != update["original_rank"] for update in unique_updates
                ),
                "discarded_weight": discarded,
                "truncation_error_budget": TRUNCATION_BUDGET,
                "rank_records": records,
                "scalability_claim_allowed": False,
                "release_gate_allowed": False,
                "command": " ".join(sys.argv),
                "commit": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
                ).strip(),
                "source_artifacts": [
                    {"kind": "single_gpu_failure", "path": str(args.single_gpu_artifact), "sha256": sha256(args.single_gpu_artifact)},
                    {"kind": "workload", "path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())},
                    {"kind": "raw_log", "path": str(args.raw_log), "sha256": sha256(args.raw_log)},
                    {"kind": "gpu_samples", "path": str(args.gpu_samples), "sha256": sha256(args.gpu_samples)},
                ],
            }
            if args.profile == "capacity":
                require_general_mps_capacity(payload)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"output": str(args.output), "status": status}), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
