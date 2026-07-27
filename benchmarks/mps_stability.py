"""Eight-rank multi-step MPS soak and checkpoint equivalence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
import torch.distributed as dist

import flagquantum as fq

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.issue092_general_mps_capacity import (  # noqa: E402
    TRUNCATION_BUDGET,
    build_entangling_workload,
)
from examples.distributed_mps.variable_bond_capacity_8gpu import (  # noqa: E402
    rank_owned_initial_mps,
)


def parameters(device):
    return (
        torch.tensor(0.07, device=device, requires_grad=True),
        torch.tensor(-0.11, device=device, requires_grad=True),
    )


def train(
    params,
    initial,
    *,
    steps,
    optimizer,
    device,
    checkpoint_dir=None,
    checkpoint_interval=25,
    resume=False,
    max_bond=16,
):
    return fq.train_distributed_mps(
        build_entangling_workload(32, *params),
        steps=steps,
        observable={16: "z"},
        optimizer=optimizer,
        lr=0.01,
        device=device,
        max_bond=max_bond,
        gradient_policy="approximate",
        gradient_tolerance=TRUNCATION_BUDGET,
        initial_mps_tensors=initial,
        initial_mps_left_canonical=True,
        checkpoint_dir=checkpoint_dir,
        checkpoint_interval=checkpoint_interval,
        resume=resume,
        memory_leak_tolerance_bytes=64 << 20,
        memory_warmup_steps=5,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimizer", choices=("sgd", "adam"), required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--restart-steps", type=int, default=10)
    parser.add_argument("--max-bond", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    if world != 8:
        raise SystemExit("ISSUE-093 soak requires eight ranks")
    torch.manual_seed(930_052 + rank)
    initial = rank_owned_initial_mps(32, 8, device)
    soak_params = parameters(device)
    soak = train(
        soak_params,
        initial,
        steps=args.steps,
        optimizer=args.optimizer,
        device=device,
        max_bond=args.max_bond,
    )

    uninterrupted_params = parameters(device)
    uninterrupted = train(
        uninterrupted_params,
        initial,
        steps=args.restart_steps,
        optimizer=args.optimizer,
        device=device,
        max_bond=args.max_bond,
    )
    root = Path(tempfile.gettempdir()) / f"fq-issue093-{args.optimizer}"
    if rank == 0:
        shutil.rmtree(root, ignore_errors=True)
    dist.barrier()
    split_params = parameters(device)
    split = train(
        split_params,
        initial,
        steps=args.restart_steps // 2,
        optimizer=args.optimizer,
        device=device,
        checkpoint_dir=root,
        checkpoint_interval=args.restart_steps // 2,
        max_bond=args.max_bond,
    )
    resumed_params = parameters(device)
    resumed = train(
        resumed_params,
        initial,
        steps=args.restart_steps,
        optimizer=args.optimizer,
        device=device,
        checkpoint_dir=root,
        checkpoint_interval=args.restart_steps // 2,
        resume=True,
        max_bond=args.max_bond,
    )
    parameter_error = max(
        abs(float(left.detach()) - float(right.detach()))
        for left, right in zip(uninterrupted_params, resumed_params)
    )
    loss_error = abs(uninterrupted.losses[-1] - resumed.losses[-1])
    local = {
        "rank": rank,
        "optimizer": args.optimizer,
        "completed_steps": soak.completed_steps,
        "memory_growth_bytes": soak.memory_growth_bytes,
        "suspected_memory_leak": soak.suspected_memory_leak,
        "memory_timeline": [
            {
                "step": step.step,
                "allocated": step.allocated_memory_bytes,
                "reserved": step.reserved_memory_bytes,
                "peak": step.peak_memory_bytes,
                "tape": step.tape_memory_bytes,
                "optimizer": step.optimizer_memory_bytes,
                "communication": step.communication_buffer_bytes,
            }
            for step in soak.steps
        ],
        "checkpoint_contract_fingerprint": soak.checkpoint_contract_fingerprint,
        "checkpoint_files": split.checkpoint_files + resumed.checkpoint_files,
        "restart_start_step": resumed.start_step,
        "restart_parameter_error": parameter_error,
        "restart_loss_error": loss_error,
        "rank_useful_work": soak.summary()["rank_useful_work"],
    }
    records = [None] * world
    dist.all_gather_object(records, local)
    if rank == 0:
        payload = {
            "schema": "flagquantum.mps_stability_run.v1",
            "world_size": world,
            "optimizer": args.optimizer,
            "steps": args.steps,
            "max_bond": args.max_bond,
            "warmup_steps": 5,
            "memory_growth_tolerance_bytes": 64 << 20,
            "rank_records": records,
            "restart_equivalent": all(
                record["restart_parameter_error"] <= 1e-7
                and record["restart_loss_error"] <= 1e-7
                and record["restart_start_step"] == args.restart_steps // 2
                for record in records
            ),
            "memory_stable": all(
                not record["suspected_memory_leak"] for record in records
            ),
            "all_ranks_useful": all(record["rank_useful_work"] for record in records),
            "full_mps_materialization": False,
            "scalability_claim_allowed": False,
            "release_gate_allowed": False,
            "command": " ".join(sys.argv),
            "commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
            ).strip(),
            "workload_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(
            json.dumps(
                {
                    "output": str(args.output),
                    "memory_stable": payload["memory_stable"],
                    "restart_equivalent": payload["restart_equivalent"],
                }
            ),
            flush=True,
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
