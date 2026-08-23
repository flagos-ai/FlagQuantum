from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist

import flagquantum as fq


def workload(device: torch.device, n_wires: int, value: float = 0.23):
    theta = torch.tensor(
        value, dtype=torch.float64, device=device, requires_grad=True
    )
    circuit = fq.Circuit(n_wires, dtype="complex128", device=device)
    for wire in range(n_wires):
        circuit.ry(wire, theta)
    for wire in range(n_wires - 1):
        if wire % 2 == 0:
            circuit.rxx(wire, wire + 1, theta)
        else:
            circuit.rzz(wire, wire + 1, theta)
    return circuit, theta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimizer", choices=("sgd", "adam"), required=True)
    parser.add_argument("--cycles", type=int, default=10)
    parser.add_argument("--backend", choices=("gloo", "nccl"), default="gloo")
    parser.add_argument("--n-wires", type=int, default=4)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.cycles != 10:
        raise ValueError("production recovery contract requires exactly 10 cycles")

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
        reference_circuit, reference_parameter = workload(device, args.n_wires)
        reference = fq.train_distributed_mps(
            reference_circuit,
            steps=args.cycles,
            optimizer=args.optimizer,
            lr=0.01,
            initial_bond_dimension=2,
            compile_site_kernels=False,
        )

        restarted_losses: list[float] = []
        restarted_parameter = None
        for completed_steps in range(1, args.cycles + 1):
            restarted_circuit, restarted_parameter = workload(device, args.n_wires)
            restarted = fq.train_distributed_mps(
                restarted_circuit,
                steps=completed_steps,
                optimizer=args.optimizer,
                lr=0.01,
                initial_bond_dimension=2,
                compile_site_kernels=False,
                checkpoint_dir=root,
                checkpoint_retention_generations=None,
                resume=completed_steps > 1,
            )
            assert restarted.start_step == completed_steps - 1
            assert len(restarted.losses) == 1
            restarted_losses.extend(restarted.losses)

        torch.testing.assert_close(
            torch.tensor(restarted_losses),
            torch.tensor(reference.losses),
            rtol=1e-10,
            atol=1e-10,
        )
        assert restarted_parameter is not None
        torch.testing.assert_close(
            restarted_parameter.detach(),
            reference_parameter.detach(),
            rtol=1e-10,
            atol=1e-10,
        )

        manifest = json.loads((root / "COMMITTED.json").read_text("utf-8"))
        assert manifest["completed_steps"] == args.cycles
        assert len(manifest["shards"]) == dist.get_world_size()
        assert len(list(root.glob("rank-*-step-*.pt"))) == (
            args.cycles * dist.get_world_size()
        )

        if rank == 0:
            stale = root / "rank-0-step-1.pt"
            data = bytearray(stale.read_bytes())
            data[len(data) // 2] ^= 0xFF
            stale.write_bytes(data)
        dist.barrier()
        latest_circuit, latest_parameter = workload(device, args.n_wires)
        latest = fq.train_distributed_mps(
            latest_circuit,
            steps=args.cycles,
            optimizer=args.optimizer,
            lr=0.01,
            initial_bond_dimension=2,
            compile_site_kernels=False,
            checkpoint_dir=root,
            checkpoint_retention_generations=None,
            resume=True,
        )
        assert latest.start_step == args.cycles
        torch.testing.assert_close(
            latest_parameter.detach(),
            reference_parameter.detach(),
            rtol=1e-10,
            atol=1e-10,
        )
        rank_summaries = [None] * dist.get_world_size()
        dist.all_gather_object(rank_summaries, restarted.summary())
        if rank == 0:
            payload = {
                        "schema": "flagquantum.mps_checkpoint_restart_cycles.v1",
                        "optimizer": args.optimizer,
                        "cycles": args.cycles,
                        "n_wires": args.n_wires,
                        "backend": args.backend,
                        "device_type": device.type,
                        "device_name": (
                            torch.cuda.get_device_name(device)
                            if device.type == "cuda"
                            else "cpu"
                        ),
                        "world_size": dist.get_world_size(),
                        "distribution_semantics": "sharded_across_ranks",
                        "checkpoint_commit_protocol": (
                            "immutable_rank_shards_atomic_manifest_v1"
                        ),
                        "loss_max_abs_error": max(
                            abs(actual - expected)
                            for actual, expected in zip(
                                restarted_losses, reference.losses
                            )
                        ),
                        "parameter_max_abs_error": float(
                            (latest_parameter - reference_parameter)
                            .detach()
                            .abs()
                            .max()
                        ),
                        "stale_generation_isolation": True,
                        "rank_summaries": rank_summaries,
                        "release_gate_allowed": False,
                        "blockers": (
                            ["accelerator_restart_development_evidence_only"]
                            if device.type == "cuda"
                            else [
                                "cpu_semantic_evidence_only",
                                "accelerator_restart_evidence_not_attached",
                            ]
                        ),
                    }
            text = json.dumps(payload, sort_keys=True)
            print(text, flush=True)
            if args.output is not None:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(text + "\n", encoding="utf-8")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
