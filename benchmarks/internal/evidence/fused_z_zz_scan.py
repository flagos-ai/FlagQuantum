"""Numerical and launch-count evidence for the fused Z/ZZ scan pair."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist

import flagquantum as fq
from flagquantum.runtime.backends.mps.reverse import execute_torch_distributed_mps_reverse
from flagquantum.simulation.mps_site_kernels import (
    reset_site_kernel_stats,
    site_kernel_stats,
)
from flagquantum.simulation.mps_execution import run_mps


def workload(parameters, *, device, n_wires=8):
    theta, phi = parameters
    circuit = fq.Circuit(n_wires, device=device)
    for wire in range(n_wires):
        circuit.ry(wire, theta if wire % 2 == 0 else phi)
    return circuit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("gloo", "nccl"), default="nccl")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compile-observables", action="store_true")
    args = parser.parse_args()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if args.backend == "nccl":
        torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank) if args.backend == "nccl" else torch.device("cpu")
    dist.init_process_group(args.backend, device_id=device if args.backend == "nccl" else None)
    rank, world = dist.get_rank(), dist.get_world_size()
    n_wires = 8 * world
    parameters = (
        torch.tensor(0.23, device=device, requires_grad=True),
        torch.tensor(-0.31, device=device, requires_grad=True),
    )
    sites = tuple(range(n_wires))
    targets = torch.sin(
        torch.arange(1, 2 * n_wires, device=device, dtype=torch.float32) * 0.73
    ) * 0.31
    terms = tuple(({wire: "z"}, targets[wire]) for wire in sites) + tuple(
        ({wire: "z", wire + 1: "z"}, targets[n_wires + wire])
        for wire in range(n_wires - 1)
    )
    reset_site_kernel_stats()
    result = execute_torch_distributed_mps_reverse(
        workload(parameters, device=device, n_wires=n_wires),
        observable_terms=terms, device=device, gradient_policy="exact",
        compile_observables=args.compile_observables,
    )
    result.backward()
    actual_value = float(result.value)
    actual_gradients = tuple(float(value.grad) for value in parameters)
    kernel_stats = site_kernel_stats()
    transfer_calls = int(kernel_stats["transfer_calls"])
    observations = [None] * world
    dist.all_gather_object(observations, {
        "rank": rank, "value": actual_value, "gradients": actual_gradients,
        "transfer_calls": transfer_calls, "summary": result.summary(),
        "kernel_stats": kernel_stats,
    })
    if rank == 0:
        reference_parameters = tuple(value.detach().clone().requires_grad_(True) for value in parameters)
        reference_state = run_mps(
            workload(reference_parameters, device=device, n_wires=n_wires)
        )
        z_values, zz_values = reference_state.expectation_z_and_nearest_neighbor_zz(sites)
        values = torch.cat((z_values, zz_values), dim=-1)
        reference_loss = (values - targets.unsqueeze(0)).square().mean()
        reference_loss.backward()
        reference_gradients = tuple(float(value.grad) for value in reference_parameters)
        max_value_error = abs(actual_value - float(reference_loss.detach()))
        max_gradient_error = max(abs(a - b) for a, b in zip(actual_gradients, reference_gradients))
        # Legacy objective: left/right value scans, per-term insertion, then
        # norm/Z/accumulator reverse scan. The fused automaton performs norm,
        # previous-Z, one vectorized channel transfer, and one ZZ insertion.
        max_local_sites = (n_wires + world - 1) // world
        legacy_transfer_calls = 9 * max_local_sites
        measured_max_calls = max(item["transfer_calls"] for item in observations)
        reduction = 1.0 - measured_max_calls / legacy_transfer_calls
        payload = {
            "schema": "flagquantum.issue098.fused_z_zz_scan.v1",
            "backend": args.backend, "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
            "world_size": world, "n_wires": n_wires, "observable_terms": len(terms),
            "compiled_observables": args.compile_observables,
            "site_kernel_stats_by_rank": [
                {"rank": item["rank"], **item["kernel_stats"]}
                for item in observations
            ],
            "objective_scan_pairs": 1,
            "forward_scan_messages_per_rank_chain": world - 1,
            "reverse_scan_messages_per_rank_chain": world - 1,
            "legacy_environment_transfer_calls": legacy_transfer_calls,
            "measured_max_environment_transfer_calls_per_rank": measured_max_calls,
            "environment_transfer_launch_reduction": reduction,
            "max_value_error": max_value_error,
            "max_gradient_error": max_gradient_error,
            "distributed_gradients": actual_gradients,
            "reference_gradients": reference_gradients,
            "rank_records": observations,
            "passed": max_value_error <= 1e-5 and max_gradient_error <= 2e-4 and reduction >= 0.35,
            "performance_claim_allowed": False,
            "scalability_claim_allowed": False,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"output": str(args.output), "passed": payload["passed"]}))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
