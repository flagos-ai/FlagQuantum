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
import flagquantum.experimental.distributed as fqxd
import flagquantum.runtime.executors.mps.records as fqxm

REPO_ROOT = Path(__file__).resolve().parents[3]
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

PROFILES = {
    "smoke": (32, 8, 7),
    "capacity": (12_288, 512, 511),
    # 107.84 GiB logical batch-one MPS: beyond one 80 GiB A800 while
    # retaining roughly 13.48 GiB of rank-owned state on eight GPUs.
    "a800_capacity": (12_288, 768, 768),
}
#: Frozen geometry of the entangling workload: how many slices its gates are
#: laid out for. This is workload identity, not deployment -- the circuit places
#: one ``ry`` per slice and one two-qubit gate per interior slice boundary, so a
#: run at a different value trains a different circuit. ``mps_release_evidence``
#: patches this module constant from the manifest's ``target_world_size`` for
#: exactly that reason, and the value travels into ``topology_fingerprint`` so
#: that two runs of different geometry can never be paired as one claim.
TARGET_WORLD = 8
TRUNCATION_BUDGET = ISSUE091_TRUNCATION_BUDGET


def capacity_baseline_worlds(target_world: int) -> tuple[int, ...]:
    """Return the rank counts below ``target_world`` a baseline may be run at.

    These are the powers of two under the completion: one rank always, then each
    halving up to it. The rank boundaries nest -- the cuts of a narrower world
    are a subset of a wider one's -- so each point shards the identical frozen
    circuit rather than a rescaled variant, which is what lets a series of them
    be read as one workload against a varying rank count.

    Every point here may be measured and sealed. Which of them the release gate
    *requires* is the narrower question ``capacity_gate_failure_worlds`` answers.
    """
    if target_world <= 0:
        raise ValueError("target_world must be positive")
    worlds, world = [1], 2
    while world < target_world:
        worlds.append(world)
        world *= 2
    return tuple(worlds)


def capacity_gate_failure_worlds(target_world: int) -> tuple[int, ...]:
    """Return the rank counts whose measured failure the capacity gate requires.

    The gate asks for one rank and for half the completion's ranks: the first
    shows the workload needs sharding at all, the second shows it still does not
    fit at half the width it completed on. The intermediate points of a series
    are evidence about how capacity moves, not additional conditions, so adding
    them to a series must not retroactively block a completion that already
    paired the two the gate names.
    """
    baselines = capacity_baseline_worlds(target_world)
    if len(baselines) == 1:
        return baselines
    return (1, baselines[-1])


#: Baselines for the default frozen geometry, kept as a module constant because
#: the launcher and the pairing tests both read it.
CAPACITY_BASELINE_WORLDS = capacity_baseline_worlds(TARGET_WORLD)


def reverse_checkpoint_capacity_bytes(logical_bytes: int, world_size: int) -> int:
    """Budget exact before/after rank-owned tape state plus workspace."""
    if logical_bytes <= 0 or world_size <= 0:
        raise ValueError("logical_bytes and world_size must be positive")
    return 2 * (logical_bytes // world_size) + (1 << 30)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def target_boundaries(
    n_sites: int, target_world: int = TARGET_WORLD
) -> tuple[int, ...]:
    """Return the interior site cuts of the frozen geometry.

    These are the left sites of the ``target_world - 1`` interior boundaries, so
    a geometry of one slice has none.
    """
    if target_world <= 0:
        raise ValueError("target_world must be positive")
    return tuple(
        (rank + 1) * n_sites // target_world - 1 for rank in range(target_world - 1)
    )


def build_entangling_workload(
    n_sites: int, theta, phi, target_world: int = TARGET_WORLD
) -> fq.Circuit:
    circuit = fq.Circuit(n_sites, device=theta.device)
    for rank in range(target_world):
        qubit = (2 * rank + 1) * n_sites // (2 * target_world)
        circuit.ry(qubit, theta if rank % 2 == 0 else phi)
    for index, left in enumerate(target_boundaries(n_sites, target_world)):
        if index % 2 == 0:
            circuit.rxx(left, left + 1, phi)
        else:
            circuit.rzz(left, left + 1, theta)
    return circuit


def capacity_gradient_contract(profile: str) -> tuple[str, float]:
    """Pin the gradient contract for rank-growing entangling gates."""
    if profile in {"capacity", "a800_capacity"}:
        return "approximate", TRUNCATION_BUDGET
    return "approximate", TRUNCATION_BUDGET


def cleanup_within_budget(
    allocated_bytes: int, baseline_bytes: int, peak_bytes: int
) -> tuple[bool, int]:
    """Allow bounded CUDA runtime residue while rejecting tensor leaks."""
    limit = baseline_bytes + min(64 << 20, peak_bytes // 1000)
    return allocated_bytes <= limit, limit


def topology_fingerprint(
    n_sites: int, max_bond: int, target_world: int = TARGET_WORLD
) -> str:
    """Identify the frozen workload, including the geometry its gates follow.

    The geometry is part of the identity so that a baseline and a completion can
    only be paired when they sharded the same circuit: a run laid out for a
    different slice count carries different gates and must not be divided into
    one capacity claim.
    """
    content = {
        "n_sites": n_sites,
        "bonds": bond_dimensions(n_sites, max_bond),
        "target_world_size": target_world,
        "target_boundaries": target_boundaries(n_sites, target_world),
    }
    return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()


def capacity_baseline_payload(
    *,
    world_size: int,
    status: str,
    n_sites: int,
    initial_bond: int,
    trained_bond: int,
    logical_bytes: int,
    rank_records: list[dict],
    target_world: int = TARGET_WORLD,
) -> dict:
    """Build a measured capacity baseline without promoting an expected OOM."""
    baselines = capacity_baseline_worlds(target_world)
    if world_size not in baselines:
        raise ValueError(
            f"capacity baseline world size must be one of {baselines}, "
            f"got {world_size}"
        )
    statuses = tuple(record.get("status") for record in rank_records)
    failed_from_capacity = (
        "cuda_oom" in statuses
        and all(status in {"passed", "cuda_oom"} for status in statuses)
    )
    return {
        "schema": "flagquantum.issue092.mps_capacity_baseline.v2",
        "legacy_schema": (
            "flagquantum.issue092.single_gpu_capacity_failure.v1"
            if world_size == 1
            else None
        ),
        "status": status,
        "world_size": world_size,
        "batch_size": 1,
        "n_sites": n_sites,
        "initial_max_bond": initial_bond,
        "trained_max_bond": trained_bond,
        "logical_mps_bytes": logical_bytes,
        "device_total_memory_bytes": rank_records[0].get(
            "device_total_memory_bytes", 0
        ),
        "target_world_size": target_world,
        "topology_fingerprint": topology_fingerprint(
            n_sites, initial_bond, target_world
        ),
        "capacity_failure": failed_from_capacity,
        "single_gpu_capacity_failure": (
            failed_from_capacity if world_size == 1 else False
        ),
        "rank_records": rank_records,
        "rank_record": rank_records[0] if world_size == 1 else None,
        "scalability_claim_allowed": False,
        "release_gate_allowed": False,
    }


def validate_capacity_baseline(
    baseline: dict,
    *,
    expected_world_size: int,
    n_sites: int,
    initial_bond: int,
    logical_bytes: int,
    target_world: int = TARGET_WORLD,
) -> None:
    """Reject unmatched, synthetic, or non-capacity baseline artifacts."""
    schema = baseline.get("schema")
    supported_schema = schema == "flagquantum.issue092.mps_capacity_baseline.v2"
    legacy_single = (
        expected_world_size == 1
        and schema == "flagquantum.issue092.single_gpu_capacity_failure.v1"
    )
    capacity_failure = baseline.get("capacity_failure")
    if legacy_single:
        capacity_failure = baseline.get("single_gpu_capacity_failure")
    if (
        not (supported_schema or legacy_single)
        or int(baseline.get("world_size", expected_world_size)) != expected_world_size
        or capacity_failure is not True
        or baseline.get("topology_fingerprint")
        != topology_fingerprint(n_sites, initial_bond, target_world)
        or int(baseline.get("logical_mps_bytes", 0)) != logical_bytes
    ):
        raise ValueError(
            f"{expected_world_size}-GPU artifact does not match this workload "
            "or does not contain a measured capacity failure"
        )


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
    parser.add_argument(
        "--four-gpu-artifact",
        type=Path,
        help="matched measured four-GPU capacity-failure artifact",
    )
    parser.add_argument("--raw-log", type=Path)
    parser.add_argument("--gpu-samples", type=Path)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument(
        "--target-world",
        type=int,
        default=TARGET_WORLD,
        help=(
            "frozen geometry the workload's gates are laid out for; this is "
            "workload identity, so every leg of one capacity series must pass "
            "the same value while WORLD_SIZE varies"
        ),
    )
    args = parser.parse_args()
    n_sites, initial_bond, trained_bond = PROFILES[args.profile]
    target_world = args.target_world
    if target_world <= 0:
        raise SystemExit("--target-world must be positive")
    baseline_worlds = capacity_baseline_worlds(target_world)
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    capacity_profile = args.profile in {"capacity", "a800_capacity"}
    allowed_worlds = (*baseline_worlds, target_world)
    if capacity_profile and world not in allowed_worlds:
        raise SystemExit(
            f"capacity profile at target world {target_world} requires one of "
            f"{allowed_worlds} ranks"
        )
    completion_run = capacity_profile and world == target_world
    if completion_run and args.single_gpu_artifact is None:
        raise SystemExit(
            f"{target_world}-rank completion requires --single-gpu-artifact"
        )
    if completion_run and (args.raw_log is None or args.gpu_samples is None):
        raise SystemExit(
            f"{target_world}-rank completion requires --raw-log and --gpu-samples"
        )
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl")
    baseline_allocated = int(torch.cuda.memory_allocated(device))
    stop = threading.Event()
    monitor = threading.Thread(target=heartbeat, args=(stop, rank, device), daemon=True)
    monitor.start()
    status, error, result = "passed", None, None
    started = time.perf_counter()
    logical_bytes = logical_mps_bytes(n_sites, initial_bond)
    gradient_policy, gradient_tolerance = capacity_gradient_contract(args.profile)
    try:
        initial = rank_owned_initial_mps(n_sites, initial_bond, device)
        angle_scale = 1e-3 if capacity_profile else 1.0
        theta = torch.tensor(0.07 * angle_scale, device=device, requires_grad=True)
        phi = torch.tensor(-0.11 * angle_scale, device=device, requires_grad=True)
        result = fqxd.train_distributed_mps(
            build_entangling_workload(n_sites, theta, phi, target_world),
            steps=args.steps,
            observable={n_sites // 2: "z"},
            optimizer="adam",
            lr=0.01,
            device=device,
            max_bond=trained_bond,
            gradient_policy=gradient_policy,
            gradient_tolerance=gradient_tolerance,
            initial_mps_tensors=initial,
            initial_mps_left_canonical=True,
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
    peak_memory = int(torch.cuda.max_memory_allocated(device))
    cleanup_verified, cleanup_limit = cleanup_within_budget(
        cleanup_allocated, baseline_allocated, peak_memory
    )
    local = {
        "rank": rank,
        "status": status,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_memory_bytes": peak_memory,
        "device_total_memory_bytes": int(
            torch.cuda.get_device_properties(device).total_memory
        ),
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
        "cleanup_verified": cleanup_verified,
        "cleanup_allocated_bytes": cleanup_allocated,
        "cleanup_baseline_bytes": baseline_allocated,
        "cleanup_limit_bytes": cleanup_limit,
    }
    if world == 1:
        records = [local]
    else:
        records = [None] * world
        dist.all_gather_object(records, local)
    if rank == 0:
        if world in baseline_worlds:
            payload = capacity_baseline_payload(
                world_size=world,
                status=status,
                n_sites=n_sites,
                initial_bond=initial_bond,
                trained_bond=trained_bond,
                logical_bytes=logical_bytes,
                rank_records=records,
                target_world=target_world,
            )
        else:
            baseline = (
                json.loads(args.single_gpu_artifact.read_text())
                if args.single_gpu_artifact is not None
                else {"single_gpu_capacity_failure": False}
            )
            try:
                validate_capacity_baseline(
                    baseline,
                    expected_world_size=1,
                    n_sites=n_sites,
                    initial_bond=initial_bond,
                    logical_bytes=logical_bytes,
                    target_world=target_world,
                )
            except ValueError as exc:
                raise SystemExit(str(exc)) from exc
            # The widest baseline below the completion. It is the one that shows
            # the workload still does not fit at half the ranks, so the gate
            # stays blocked until it is attached.
            paired_world = baseline_worlds[-1] if len(baseline_worlds) > 1 else None
            four_gpu_baseline = None
            if args.four_gpu_artifact is not None:
                four_gpu_baseline = json.loads(args.four_gpu_artifact.read_text())
                try:
                    validate_capacity_baseline(
                        four_gpu_baseline,
                        expected_world_size=paired_world,
                        n_sites=n_sites,
                        initial_bond=initial_bond,
                        logical_bytes=logical_bytes,
                        target_world=target_world,
                    )
                except ValueError as exc:
                    raise SystemExit(str(exc)) from exc
            failed = [record for record in records if record["status"] != "passed"]
            if failed:
                payload = {
                    "schema": "flagquantum.issue092.general_mps_capacity_failure.v1",
                    "world_size": world,
                    "n_sites": n_sites,
                    "initial_max_bond": initial_bond,
                    "trained_max_bond": trained_bond,
                    "logical_mps_bytes": logical_bytes,
                    "target_world_size": target_world,
                    "topology_fingerprint": topology_fingerprint(
                        n_sites, initial_bond, target_world
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
                for index, left in enumerate(
                    target_boundaries(n_sites, target_world)
                )
            ]
            discarded = sum(update["discarded_weight"] for update in unique_updates)
            payload = {
                "schema": (
                    "flagquantum.issue092.general_mps_capacity.v1"
                    if capacity_profile
                    else "flagquantum.issue092.general_mps_capacity_smoke.v1"
                ),
                "batch_size": 1,
                "n_sites": n_sites,
                "initial_max_bond": initial_bond,
                "trained_max_bond": trained_bond,
                "logical_mps_bytes": logical_bytes,
                "target_world_size": target_world,
                "topology_fingerprint": topology_fingerprint(
                    n_sites, initial_bond, target_world
                ),
                "world_size": world,
                "distribution_semantics": "sharded_across_ranks",
                "single_gpu_capacity_failure": True,
                "four_gpu_capacity_failure": four_gpu_baseline is not None,
                "matched_capacity_failure_world_sizes": (
                    [1, paired_world] if four_gpu_baseline is not None else [1]
                ),
                "capacity_gate": {
                    "passed": four_gpu_baseline is not None,
                    "required_failure_world_sizes": list(
                        capacity_gate_failure_worlds(target_world)
                    ),
                    "completion_world_size": target_world,
                    "blockers": (
                        []
                        if four_gpu_baseline is not None
                        else [
                            "matched_capacity_failure_not_attached_at_world_"
                            f"{paired_world}"
                        ]
                    ),
                },
                "single_gpu_peak_memory_bytes": baseline["rank_record"]["peak_memory_bytes"],
                "single_gpu_device_total_memory_bytes": baseline.get(
                    "device_total_memory_bytes",
                    baseline["rank_record"].get("device_total_memory_bytes", 0),
                ),
                "single_gpu_artifact": str(args.single_gpu_artifact),
                "four_gpu_artifact": (
                    None
                    if args.four_gpu_artifact is None
                    else str(args.four_gpu_artifact)
                ),
                "sharded_completion": all(record["status"] == "passed" for record in records),
                "full_mps_materialization": False,
                "boundary_evidence": boundary_evidence,
                "bond_dimension_changed": any(
                    update["kept_rank"] != update["original_rank"] for update in unique_updates
                ),
                "gradient_policy": gradient_policy,
                "discarded_weight": discarded,
                "truncation_error_budget": gradient_tolerance,
                "rank_records": records,
                "scalability_claim_allowed": False,
                "release_gate_allowed": False,
                "command": " ".join(sys.argv),
                "commit": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
                ).strip(),
                "source_artifacts": [
                    {"kind": "single_gpu_failure", "path": str(args.single_gpu_artifact), "sha256": sha256(args.single_gpu_artifact)},
                    *(
                        []
                        if args.four_gpu_artifact is None
                        else [
                            {
                                "kind": "four_gpu_failure",
                                "path": str(args.four_gpu_artifact),
                                "sha256": sha256(args.four_gpu_artifact),
                            }
                        ]
                    ),
                    {"kind": "workload", "path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())},
                    {"kind": "raw_log", "path": str(args.raw_log), "sha256": sha256(args.raw_log)},
                    {"kind": "gpu_samples", "path": str(args.gpu_samples), "sha256": sha256(args.gpu_samples)},
                ],
            }
            if capacity_profile:
                require_general_mps_capacity(payload)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"output": str(args.output), "status": status}), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
