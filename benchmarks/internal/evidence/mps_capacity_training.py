"""Beyond-one-A100 full MPS training capacity workload for ISSUE-052."""

import argparse
import json
import os
import tempfile
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist

import flagquantum as fq
import flagquantum.experimental.distributed as fqxd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--global-state-bytes", type=int, default=43 << 30)
    parser.add_argument("--n-wires", type=int, default=8)
    parser.add_argument("--bond-dimension", type=int, default=2)
    args = parser.parse_args()
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if world > 1:
        dist.init_process_group("nccl", timeout=timedelta(minutes=30))
    else:
        store = tempfile.NamedTemporaryFile(prefix="fq-mps-capacity-", delete=True)
        dist.init_process_group(
            "nccl", rank=0, world_size=1, init_method=f"file://{store.name}"
        )
    complex_bytes = torch.empty((), dtype=torch.complex64).element_size()
    per_batch_elements = (
        4 * args.bond_dimension
        + max(0, args.n_wires - 2) * 2 * args.bond_dimension**2
    )
    bsz = (args.global_state_bytes + per_batch_elements * complex_bytes - 1) // (
        per_batch_elements * complex_bytes
    )
    parameters = tuple(
        torch.tensor(0.013 + index * 0.001, device=device, requires_grad=True)
        for index in range(world)
    )
    circuit = fq.Circuit(args.n_wires, bsz=bsz, device=device)
    # Every rank owns useful fixed-gate site work; one owner also computes the
    # trainable VJP and broadcasts the owner-local optimizer update.
    for wire in range(args.n_wires):
        circuit.h(wire)
    for owner, parameter in enumerate(parameters):
        circuit.rz(owner * args.n_wires // world, parameter)
    started = time.perf_counter()
    status = "passed"
    error = None
    result = None
    try:
        result = fqxd.train_distributed_mps(
            circuit,
            steps=1,
            optimizer="adam",
            lr=0.01,
            device=device,
            max_bond=1,
            cutoff=0.0,
            memory_leak_tolerance_bytes=64 << 20,
            initial_bond_dimension=args.bond_dimension,
        )
        torch.cuda.synchronize(device)
    except torch.OutOfMemoryError as exc:
        status = "cuda_oom"
        error = str(exc).splitlines()[0]
        torch.cuda.empty_cache()
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            status = "cuda_oom"
            error = str(exc).splitlines()[0]
            torch.cuda.empty_cache()
        else:
            raise
    record = {
        "rank": rank,
        "status": status,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        "useful_mps_work": (
            False
            if result is None
            else all(item.useful_work_completed for item in result.steps)
        ),
        "training_summary": None if result is None else result.summary(),
        "error": error,
    }
    gathered = [record]
    if world > 1:
        gathered = [None] * world
        dist.all_gather_object(gathered, record)
    if rank == 0:
        single_failure = world == 1 and status == "cuda_oom"
        sharded_completion = world > 1 and all(
            item["status"] == "passed" and item["useful_mps_work"]
            for item in gathered
        )
        payload = {
            "schema": "flagquantum.issue052.mps_capacity_training.v1",
            "gate": "capacity",
            "evidence_source": "measured_runtime",
            "artifact_classification": "production_training_benchmark",
            "world_size": world,
            "device": torch.cuda.get_device_name(0),
            "device_total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
            "n_wires": args.n_wires,
            "batch_size": bsz,
            "bond_dimension": args.bond_dimension,
            "global_rank_owned_mps_bytes": bsz
            * per_batch_elements
            * complex_bytes,
            "rank_records": gathered,
            "single_gpu_capacity_failure": single_failure,
            "sharded_completion": sharded_completion,
            "general_mps_training_loop": result is not None,
            "full_mps_materialization": False,
            "capacity_gate": {
                "passed": single_failure or sharded_completion,
                "component": "single_gpu_baseline" if world == 1 else "sharded_training",
            },
            "scalability_claim_allowed": False,
            "release_gate_allowed": False,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(json.dumps(payload["capacity_gate"]))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
