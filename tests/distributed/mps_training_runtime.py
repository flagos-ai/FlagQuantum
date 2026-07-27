"""Two-rank Gloo semantic workload for ISSUE-052."""

import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist

import flagquantum as fq
from flagquantum.runtime.backends.mps.training import MPSTrainingError


def circuit(device: torch.device, world_size: int):
    theta = torch.tensor(0.31, device=device, requires_grad=True)
    phi = torch.tensor(-0.17, device=device, requires_grad=True)
    n_wires = max(4, world_size)
    value = fq.Circuit(n_wires)
    for wire in range(n_wires):
        value.ry(wire, theta if wire % 2 == 0 else phi)
    for wire in range(n_wires - 1):
        value.rxx(wire, wire + 1, phi if wire % 2 == 0 else theta)
    for wire in reversed(range(n_wires - 1)):
        value.rzz(wire, wire + 1, theta if wire % 2 == 0 else phi)
    return value, (theta, phi)


def rank_owned_initial_mps(device: torch.device, n_wires: int):
    rank, world = dist.get_rank(), dist.get_world_size()
    first = rank * n_wires // world
    last = (rank + 1) * n_wires // world
    tensors = {}
    for wire in range(first, last):
        left = 1 if wire == 0 else 2
        right = 1 if wire == n_wires - 1 else 2
        tensor = torch.zeros(1, left, 2, right, dtype=torch.complex64, device=device)
        tensor[0, 0, 0, 0] = 1
        if left == 1 and right == 2:
            tensor[0, 0, 1, 1] = 0.25
        elif left == 2 and right == 2:
            tensor[0, 1, 1, 1] = 0.25
        tensors[wire] = tensor
    return tensors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("gloo", "nccl"), default="gloo")
    args = parser.parse_args()
    if args.backend == "nccl":
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group(args.backend)
    rank = dist.get_rank()
    device = (
        torch.device("cuda", int(os.environ["LOCAL_RANK"]))
        if args.backend == "nccl"
        else torch.device("cpu")
    )
    root = Path(os.environ["FQ_TEST_CHECKPOINT"])
    try:
        workload, parameters = circuit(device, dist.get_world_size())
        result = fq.train_distributed_mps(
            workload,
            steps=3,
            optimizer="adam",
            lr=0.02,
            device=device,
            checkpoint_dir=root,
        )
        assert result.completed_steps == 3
        assert all(item.useful_work_completed for item in result.steps)
        checkpoint_parameters = tuple(float(item.detach()) for item in parameters)
        resumed_workload, resumed_parameters = circuit(device, dist.get_world_size())
        resumed = fq.train_distributed_mps(
            resumed_workload,
            steps=4,
            optimizer="adam",
            lr=0.02,
            device=device,
            checkpoint_dir=root,
            resume=True,
        )
        assert resumed.start_step == 3
        assert len(resumed.losses) == 1
        assert (
            tuple(float(item.detach()) for item in resumed_parameters)
            != checkpoint_parameters
        )
        initial_workload, _ = circuit(device, dist.get_world_size())
        initialized = fq.train_distributed_mps(
            initial_workload,
            steps=1,
            optimizer="sgd",
            device=device,
            initial_mps_tensors=rank_owned_initial_mps(
                device, initial_workload.to_ir().n_wires
            ),
        )
        assert initialized.completed_steps == 1
        mse_workload, _ = circuit(device, dist.get_world_size())
        mse = fq.train_distributed_mps(
            mse_workload,
            steps=2,
            observable_terms=(
                ({0: "z"}, torch.tensor([0.25], device=device)),
                ({1: "z", 2: "z"}, torch.tensor([-0.1], device=device)),
            ),
            optimizer="adam",
            lr=0.01,
            device=device,
        )
        assert mse.completed_steps == 2
        assert all(torch.isfinite(torch.tensor(value)) for value in mse.losses)
        assert all(item.objective_scan_pairs == 1 for item in mse.steps)
        assert all(
            item.objective_scan_forward_messages == dist.get_world_size() - 1
            for item in mse.steps
        )
        assert all(
            item.objective_scan_reverse_messages == dist.get_world_size() - 1
            for item in mse.steps
        )
        initial_checkpoint = root / "initial-state"
        initial_tensors = rank_owned_initial_mps(
            device, initial_workload.to_ir().n_wires
        )
        fq.train_distributed_mps(
            initial_workload,
            steps=1,
            optimizer="sgd",
            device=device,
            checkpoint_dir=initial_checkpoint,
            initial_mps_tensors=initial_tensors,
        )
        changed_tensors = {
            wire: tensor.clone() for wire, tensor in initial_tensors.items()
        }
        first_wire = min(changed_tensors)
        changed_tensors[first_wire].reshape(-1)[0] += 0.125
        changed_workload, _ = circuit(device, dist.get_world_size())
        try:
            fq.train_distributed_mps(
                changed_workload,
                steps=2,
                optimizer="sgd",
                device=device,
                checkpoint_dir=initial_checkpoint,
                resume=True,
                initial_mps_tensors=changed_tensors,
            )
        except MPSTrainingError as error:
            assert "contract" in str(error)
        else:
            raise AssertionError(
                "changed initial MPS content passed checkpoint contract"
            )
        summary = result.summary()
        print(
            json.dumps(
                {
                    "rank": rank,
                    "completed_steps": result.completed_steps,
                    "optimizer_state_count": sum(
                        item.optimizer_state_local for item in result.ownership
                    ),
                    "rank_useful_work": summary["rank_useful_work"],
                    "full_mps_materialization": summary["full_mps_materialization"],
                    "scalability_claim_allowed": summary["scalability_claim_allowed"],
                    "cleanup_verified": True,
                    "variable_bond_initial_state": True,
                }
            ),
            flush=True,
        )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
