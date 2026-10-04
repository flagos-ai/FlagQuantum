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
from collections.abc import Mapping
from pathlib import Path

import torch
import torch.distributed as dist

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import flagquantum as fq  # noqa: E402
import flagquantum.experimental.distributed as fqxd  # noqa: E402
import flagquantum.runtime.executors.mps.records as fqxm  # noqa: E402

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
    """Return this rank's slice of the frozen initial state.

    The state is one left-canonical tensor-train whose tensor at a wire is a
    function of that wire's bond pair alone, so every rank draws the distinct
    bond shapes in the order the whole chain visits them and then keeps the
    wires it owns. Drawing the shapes in the order this rank's own slice
    happens to visit them would give a wire a different tensor at every rank
    count: the generator advances once per newly seen shape, so a rank whose
    slice begins deep in the chain would draw the tensor that belongs to an
    earlier wire. The state would then depend on the rank boundaries, a
    single-device leg and a sharded leg of the same frozen circuit would start
    from different states, and the truncation a sharded leg reports would be a
    property of its deployment rather than of the workload. The tensors
    themselves are unchanged: the same generator, the same seed, and the same
    draw order as a whole-chain walk, which is what a single-device leg builds.
    """

    rank, world = dist.get_rank(), dist.get_world_size()
    first, last = rank * n_sites // world, (rank + 1) * n_sites // world
    bonds = bond_dimensions(n_sites, max_bond)
    generator = torch.Generator(device=device).manual_seed(520_052)
    cache: dict[tuple[int, int], torch.Tensor] = {}
    for wire in range(n_sites):
        shape = (bonds[wire], bonds[wire + 1])
        if shape in cache:
            continue
        real = torch.randn(2 * shape[0], shape[1], device=device, generator=generator)
        imag = torch.randn(2 * shape[0], shape[1], device=device, generator=generator)
        q, _ = torch.linalg.qr(torch.complex(real, imag), mode="reduced")
        cache[shape] = q.reshape(1, shape[0], 2, shape[1]).contiguous()
    return {wire: cache[(bonds[wire], bonds[wire + 1])] for wire in range(first, last)}


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


def rank_parameter_wires() -> tuple[int, ...]:
    """The wire on which each rank contributes a rotation of its own."""

    return tuple(
        (2 * rank + 1) * N_SITES // (2 * TARGET_WORLD) for rank in range(TARGET_WORLD)
    )


def parameter_count() -> int:
    """The number of independent trainable leaves the frozen circuit carries.

    One leaf per gate, which is the binding the release contract needs: the
    optimizer assigns parameter ``index`` to rank ``index % world_size``, so a
    circuit with fewer leaves than ranks leaves the surplus ranks owning no
    parameter and no optimizer state, and the per-parameter ownership the
    release payload reports is then true by construction rather than measured.
    """

    return len(rank_parameter_wires()) + len(target_boundaries())


def initial_parameter_values() -> dict[str, float]:
    """Return the starting angle of every leaf, keyed by the gate it drives.

    The values alternate exactly as the two shared scalars they replace did, so
    the forward arithmetic of the frozen circuit is unchanged by the rebinding
    and the earlier single-device measurement still describes this workload.
    """

    values = {
        f"ry:{wire}": 7e-5 if rank % 2 == 0 else -1.1e-4
        for rank, wire in enumerate(rank_parameter_wires())
    }
    values.update(
        {
            f"rxx:{left}": -1.1e-4 if index % 2 == 0 else 7e-5
            for index, left in enumerate(target_boundaries())
        }
    )
    return values


def frozen_parameters(device: torch.device) -> dict[str, torch.Tensor]:
    """Return one trainable leaf per gate of the frozen circuit."""

    return {
        name: torch.tensor(value, device=device, requires_grad=True)
        for name, value in initial_parameter_values().items()
    }


def topology_fingerprint() -> str:
    content = {
        "n_sites": N_SITES,
        "bonds": bond_dimensions(N_SITES, MAX_BOND),
        "target_world_size": TARGET_WORLD,
        "target_boundaries": target_boundaries(),
    }
    return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()


def workload(parameters: Mapping[str, torch.Tensor]) -> fq.Circuit:
    device = next(iter(parameters.values())).device
    circuit = fq.Circuit(N_SITES, device=device)
    for wire in rank_parameter_wires():
        circuit.ry(wire, parameters[f"ry:{wire}"])
    for left in target_boundaries():
        circuit.rxx(left, left + 1, parameters[f"rxx:{left}"])
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
        parameters = frozen_parameters(device)
        result = fqxd.train_distributed_mps(
            workload(parameters),
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
    if "parameters" in locals():
        del parameters
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
            "parameter_count": parameter_count(),
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
                "parameter_count": parameter_count(),
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
