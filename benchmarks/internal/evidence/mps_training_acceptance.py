"""Measured matched-workload MPS training artifacts for ISSUE-052."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import socket
import statistics
import subprocess
import tempfile
import time
from pathlib import Path

import torch
import torch.distributed as dist

import flagquantum as fq


def source_identity() -> tuple[str, bool]:
    commit = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        check=False,
        capture_output=True,
        text=True,
    ).stdout.strip()
    tracked_changes = subprocess.run(
        ("git", "diff", "--quiet"),
        check=False,
    ).returncode != 0 or subprocess.run(
        ("git", "diff", "--cached", "--quiet"),
        check=False,
    ).returncode != 0
    return commit or "unavailable", tracked_changes


def build_circuit(
    parameters: tuple[torch.Tensor, ...], n_wires: int, layers: int, family: str
):
    theta, phi = parameters
    circuit = fq.Circuit(n_wires, device=theta.device)
    for layer in range(layers):
        for wire in range(n_wires):
            if family == "tfim_time_evolution":
                circuit.rx(wire, theta)
            else:
                circuit.ry(wire, theta if (wire + layer) % 2 == 0 else phi)
        start = layer % 2
        for wire in range(start, n_wires - 1, 2):
            if family == "tfim_time_evolution":
                circuit.rzz(wire, wire + 1, phi)
            else:
                circuit.rxx(wire, wire + 1, phi if layer % 2 == 0 else theta)
        for wire in range(1 - start, n_wires - 1, 2):
            circuit.rzz(wire, wire + 1, theta if layer % 2 == 0 else phi)
    return circuit


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(fraction * (len(ordered) - 1)))]


def distributed_run(args, device):
    parameters = (
        torch.tensor(0.31, device=device, requires_grad=True),
        torch.tensor(-0.17, device=device, requires_grad=True),
    )
    result = fq.train_distributed_mps(
        build_circuit(parameters, args.n_wires, args.layers, args.family),
        steps=args.warmup + args.repetitions,
        optimizer="adam",
        lr=args.lr,
        device=device,
        max_bond=args.max_bond,
        cutoff=args.cutoff,
        memory_leak_tolerance_bytes=args.memory_leak_tolerance_bytes,
        gradient_policy=args.gradient_policy,
        gradient_tolerance=args.gradient_tolerance,
    )
    local = [item.end_to_end_seconds for item in result.steps]
    local_components = [
        {
            "forward": item.forward_seconds,
            "reverse": item.reverse_seconds,
            "optimizer": item.optimizer_seconds,
            "end_to_end": item.end_to_end_seconds,
        }
        for item in result.steps
    ]
    local_memory = [item.peak_memory_bytes for item in result.steps]
    gathered_time = [None] * dist.get_world_size()
    gathered_memory = [None] * dist.get_world_size()
    gathered_components = [None] * dist.get_world_size()
    gathered_training = [None] * dist.get_world_size()
    gathered_placement = [None] * dist.get_world_size()
    dist.all_gather_object(gathered_time, local)
    dist.all_gather_object(gathered_memory, local_memory)
    dist.all_gather_object(gathered_components, local_components)
    dist.all_gather_object(gathered_training, result.summary())
    dist.all_gather_object(
        gathered_placement,
        {
            "rank": dist.get_rank(),
            "local_rank": int(os.environ.get("LOCAL_RANK", "0")),
            "node_rank": int(os.environ.get("GROUP_RANK", "0")),
            "hostname": socket.gethostname(),
            "device": str(device),
        },
    )
    samples = [
        max(rank_values[i] for rank_values in gathered_time) for i in range(len(local))
    ]
    memories = [
        max(rank_values[i] for rank_values in gathered_memory)
        for i in range(len(local))
    ]
    components = [
        {
            name: max(rank_values[i][name] for rank_values in gathered_components)
            for name in ("forward", "reverse", "optimizer", "end_to_end")
        }
        for i in range(len(local))
    ]
    return (
        samples[args.warmup :],
        memories[args.warmup :],
        list(result.losses[args.warmup :]),
        result.summary(),
        components[args.warmup :],
        gathered_memory,
        gathered_training,
        gathered_placement,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n-wires", type=int, default=16)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--max-bond", type=int, default=64)
    parser.add_argument("--cutoff", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--memory-leak-tolerance-bytes", type=int, default=16 << 20)
    parser.add_argument(
        "--family",
        choices=("tfim_time_evolution", "circuit_training", "variable_bond_training"),
        default="circuit_training",
    )
    parser.add_argument("--scaling-mode", choices=("strong", "weak"), default="strong")
    parser.add_argument("--gradient-policy", choices=("exact", "approximate"), default="exact")
    parser.add_argument("--gradient-tolerance", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=520052)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if world_size > 1:
        dist.init_process_group("nccl")
    else:
        store = tempfile.NamedTemporaryFile(prefix="fq-mps-benchmark-", delete=True)
        dist.init_process_group(
            "nccl",
            rank=0,
            world_size=1,
            init_method=f"file://{store.name}",
        )
    try:
        (
            samples,
            memories,
            losses,
            training,
            components,
            rank_memory_samples,
            rank_training,
            rank_placement,
        ) = distributed_run(args, device)
        if world_size > 1 and dist.get_rank() != 0:
            return
        mean = statistics.mean(samples)
        stdev = statistics.stdev(samples)
        margin = 2.093 * stdev / len(samples) ** 0.5
        workload = {
            "n_wires": args.n_wires,
            "layers": args.layers,
            "max_bond": args.max_bond,
            "cutoff": args.cutoff,
            "optimizer": "adam",
            "learning_rate": args.lr,
            "family": args.family,
            "scaling_mode": args.scaling_mode,
            "seed": args.seed,
        }
        local_memory_bytes_by_rank = [
            max(int(value) for value in values) for values in rank_memory_samples
        ]
        communication_bytes_by_rank = [
            sum(
                int(step["boundary_bytes"])
                + int(step["gradient_collective_bytes"])
                + int(step["optimizer_collective_bytes"])
                for step in summary["step_metrics"][args.warmup :]
            )
            for summary in rank_training
        ]
        source_commit, source_tree_dirty = source_identity()
        payload = {
            "schema": "flagquantum.issue052.mps_training_measurement.v1",
            "benchmark": "mps_single_node_training_certification",
            "evidence_source": "measured_runtime",
            "measured": True,
            "source_commit": source_commit,
            "source_tree_dirty": source_tree_dirty,
            "timestamp": time.time(),
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(0),
            "world_size": world_size,
            "local_world_size": int(os.environ.get("LOCAL_WORLD_SIZE", world_size)),
            "node_count": len({item["hostname"] for item in rank_placement}),
            "rank_placement": rank_placement,
            "local_memory_bytes_by_rank": local_memory_bytes_by_rank,
            "communication_bytes": max(communication_bytes_by_rank, default=0),
            "communication_bytes_by_rank": communication_bytes_by_rank,
            "communication_protocol": (
                "none"
                if world_size == 1
                else "batched_isend_irecv_and_owner_gradient_collectives"
            ),
            "claim_evidence_type": "development_smoke",
            "benchmark_evidence_class": "non_release_smoke",
            "non_release_evidence": True,
            "scalability_blockers": [
                "development_smoke_not_release_scalability_evidence"
            ],
            "distribution_semantics": (
                "single_device_fast_path" if world_size == 1 else "sharded_across_ranks"
            ),
            "workload": workload,
            "family": args.family,
            "scaling_mode": args.scaling_mode,
            "workload_sha256": hashlib.sha256(
                json.dumps(workload, sort_keys=True).encode()
            ).hexdigest(),
            "warmup": args.warmup,
            "repetitions": args.repetitions,
            "samples_seconds": samples,
            "mean_seconds": mean,
            "confidence_interval_95_seconds": [mean - margin, mean + margin],
            "p50_seconds": percentile(samples, 0.5),
            "p95_seconds": percentile(samples, 0.95),
            "coefficient_of_variation": stdev / mean,
            "component_samples_seconds": components,
            "peak_memory_bytes_samples": memories,
            "peak_memory_bytes": max(memories),
            "memory_growth_bytes": max(0, memories[-1] - memories[0]),
            "losses": losses,
            "completed_work_units": len(samples),
            "rank_useful_work": True
            if training is None
            else training["rank_useful_work"],
            "full_mps_materialization": False,
            "training": training,
            "scalability_claim_allowed": False,
            "release_gate_allowed": False,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(
            json.dumps(
                {
                    "output": str(args.output),
                    "mean_seconds": mean,
                    "world_size": world_size,
                }
            )
        )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
