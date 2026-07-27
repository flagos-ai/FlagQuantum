"""Measured MPS-shaped storage boundary probe; never general training evidence."""

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--global-bytes", type=int, default=48 << 30)
    args = parser.parse_args()
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    if world > 1:
        dist.init_process_group("nccl")
    device = torch.device("cuda", local_rank)
    local_bytes = (args.global_bytes + world - 1) // world
    element_size = torch.empty((), dtype=torch.complex64).element_size()
    elements = (local_bytes + element_size - 1) // element_size
    status = "passed"
    error = None
    useful_work = False
    peak = 0
    started = time.perf_counter()
    try:
        tensor = torch.empty(elements, dtype=torch.complex64, device=device)
        # Touch one value per 2 MiB page plus the final value so the allocator
        # and kernel path are both real without timing a 48 GiB memory sweep.
        stride = max(1, (2 << 20) // element_size)
        tensor[::stride] = complex(rank + 1, 0)
        tensor[-1] = complex(rank + 1, 0)
        torch.cuda.synchronize(device)
        useful_work = True
        peak = int(torch.cuda.max_memory_allocated(device))
        del tensor
    except torch.OutOfMemoryError as exc:
        status = "cuda_oom"
        error = str(exc).splitlines()[0]
        torch.cuda.empty_cache()
    elapsed = time.perf_counter() - started
    local = {
        "rank": rank,
        "status": status,
        "requested_local_bytes": local_bytes,
        "peak_memory_bytes": peak,
        "useful_tensor_work": useful_work,
        "elapsed_seconds": elapsed,
        "error": error,
    }
    gathered = [None] * world
    if world > 1:
        dist.all_gather_object(gathered, local)
    else:
        gathered = [local]
    if rank == 0:
        payload = {
            "schema": "flagquantum.issue052.mps_capacity_probe.v1",
            "gate": "capacity",
            "evidence_source": "measured_runtime",
            "artifact_classification": "measured_accelerator_probe",
            "world_size": world,
            "device": torch.cuda.get_device_name(0),
            "device_total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
            "global_mps_shaped_bytes": args.global_bytes,
            "rank_records": gathered,
            "single_gpu_capacity_failure": world == 1 and status == "cuda_oom",
            "sharded_completion": world > 1 and all(item["status"] == "passed" for item in gathered),
            "full_mps_materialization": False,
            "general_mps_training_loop": False,
            "capacity_gate": {
                "passed": False,
                "reason": "storage_boundary_probe_is_not_general_mps_training",
            },
            "scalability_claim_allowed": False,
            "release_gate_allowed": False,
            "blockers": ["general_mps_capacity_training_not_executed"],
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(json.dumps(payload["capacity_gate"]))
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
