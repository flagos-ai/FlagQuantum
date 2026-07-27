"""Seeded distributed-MPS correctness case for ISSUE-091."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import torch
import torch.distributed as dist

import flagquantum as fq
from flagquantum.runtime.backends.mps.reverse import execute_torch_distributed_mps_reverse
from flagquantum.simulation.mps import run_mps

SEED = 91_052
TOLERANCES = {
    "value_atol": 5e-5,
    "gradient_atol": 3e-4,
    "parameter_atol": 3e-5,
    "directional_atol": 5e-4,
    "approximate_value_atol": 1e-4,
    "approximate_gradient_atol": 8e-4,
}


def workload(parameters, *, device, dtype):
    theta, phi = parameters
    circuit = fq.Circuit(8, device=device, dtype=dtype)
    for wire in range(8):
        circuit.ry(wire, theta if wire % 2 == 0 else phi)
    for wire in range(7):
        circuit.rxx(wire, wire + 1, phi if wire % 2 == 0 else theta)
    for wire in reversed(range(7)):
        circuit.rzz(wire, wire + 1, theta if wire % 2 == 0 else phi)
    circuit.rx(3, theta)
    circuit.ry(4, phi)
    return circuit


def dense_loss(parameters, *, device, dtype):
    circuit = workload(parameters, device=device, dtype=dtype)
    state = circuit.state(refresh=True)
    indices = torch.arange(state.shape[-1], device=device)
    sign = torch.ones(state.shape[-1], device=device)
    for wire in (1, 6):
        bit = (indices >> (7 - wire)) & 1
        sign *= 1 - 2 * bit
    return torch.sum(state.abs().square() * sign, dim=-1).mean()


def optimizer_for(name, parameters):
    return (
        torch.optim.Adam(parameters, lr=0.01)
        if name == "adam"
        else torch.optim.SGD(parameters, lr=0.01)
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("gloo", "nccl"), default="gloo")
    parser.add_argument("--optimizer", choices=("sgd", "adam"), default="adam")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument(
        "--dtype", choices=("complex64", "complex128"), default="complex64"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if args.backend == "nccl":
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    dtype = getattr(torch, args.dtype)
    real_dtype = torch.float64 if dtype == torch.complex128 else torch.float32
    dist.init_process_group(args.backend)
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.manual_seed(SEED)
    parameters = (
        torch.tensor(0.23, device=device, dtype=real_dtype, requires_grad=True),
        torch.tensor(-0.31, device=device, dtype=real_dtype, requires_grad=True),
    )
    owners = tuple(index % world for index in range(2))
    owned = [parameter for parameter, owner in zip(parameters, owners) if owner == rank]
    distributed_optimizer = optimizer_for(args.optimizer, owned) if owned else None
    dense_parameters = None
    dense_optimizer = None
    local_parameters = None
    local_optimizer = None
    if rank == 0:
        dense_parameters = tuple(
            value.detach().clone().requires_grad_(True) for value in parameters
        )
        local_parameters = tuple(
            value.detach().clone().requires_grad_(True) for value in parameters
        )
        dense_optimizer = optimizer_for(args.optimizer, list(dense_parameters))
        local_optimizer = optimizer_for(args.optimizer, list(local_parameters))
    step_records = []
    first_gradients = None
    exact_discarded = 0.0
    for step in range(args.steps):
        for parameter in parameters:
            parameter.grad = None
        result = execute_torch_distributed_mps_reverse(
            workload(parameters, device=device, dtype=dtype),
            observable={1: "z", 6: "z"},
            device=device,
            dtype=dtype,
            max_bond=None,
            cutoff=0.0,
            gradient_policy="exact",
        )
        result.backward()
        exact_discarded = max(exact_discarded, float(result.discarded_weight))
        gradients = tuple(float(parameter.grad.detach()) for parameter in parameters)
        observations: list[object] = [None] * world
        dist.all_gather_object(
            observations,
            {"rank": rank, "value": float(result.value), "gradients": gradients},
        )
        if first_gradients is None:
            first_gradients = gradients
        record = {"step": step, "distributed_value": float(result.value)}
        if rank == 0:
            assert dense_parameters is not None and local_parameters is not None
            assert dense_optimizer is not None and local_optimizer is not None
            dense_optimizer.zero_grad()
            local_optimizer.zero_grad()
            dense_value = dense_loss(dense_parameters, device=device, dtype=dtype)
            dense_value.backward()
            local_state = run_mps(
                workload(local_parameters, device=device, dtype=dtype),
                max_bond=None,
                cutoff=0.0,
            )
            local_value = local_state.expectation_ps(z=(1, 6)).mean()
            local_value.backward()
            record.update(
                {
                    "dense_value": float(dense_value.detach()),
                    "local_mps_value": float(local_value.detach()),
                    "value_error": abs(
                        float(result.value) - float(dense_value.detach())
                    ),
                    "local_value_error": abs(
                        float(result.value) - float(local_value.detach())
                    ),
                    "gradients": list(gradients),
                    "dense_gradients": [
                        float(value.grad) for value in dense_parameters
                    ],
                    "local_mps_gradients": [
                        float(value.grad) for value in local_parameters
                    ],
                    "gradient_errors": [
                        abs(gradient - float(reference.grad))
                        for gradient, reference in zip(gradients, dense_parameters)
                    ],
                    "rank_observations": observations,
                }
            )
            dense_optimizer.step()
            local_optimizer.step()
        if distributed_optimizer is not None:
            distributed_optimizer.step()
        for parameter, owner in zip(parameters, owners):
            dist.broadcast(parameter.data, src=owner)
        if rank == 0:
            record["parameters"] = [float(value.detach()) for value in parameters]
            record["dense_parameters"] = [
                float(value.detach()) for value in dense_parameters
            ]
            record["parameter_errors"] = [
                abs(float(value.detach()) - float(reference.detach()))
                for value, reference in zip(parameters, dense_parameters)
            ]
            step_records.append(record)

    approximate_parameters = tuple(
        value.detach().clone().requires_grad_(True) for value in parameters
    )
    approximate = execute_torch_distributed_mps_reverse(
        workload(approximate_parameters, device=device, dtype=dtype),
        observable={1: "z", 6: "z"},
        device=device,
        dtype=dtype,
        max_bond=2,
        gradient_policy="approximate",
        gradient_tolerance=100.0,
    )
    approximate.backward()
    approximate_budget = 0.1
    directional_error = 0.0
    if rank == 0:
        assert first_gradients is not None
        epsilon = 1e-3
        direction = (0.6, -0.8)
        plus = tuple(
            torch.tensor(base + epsilon * delta, device=device, dtype=real_dtype)
            for base, delta in zip((0.23, -0.31), direction)
        )
        minus = tuple(
            torch.tensor(base - epsilon * delta, device=device, dtype=real_dtype)
            for base, delta in zip((0.23, -0.31), direction)
        )
        finite_difference = float(
            (
                dense_loss(plus, device=device, dtype=dtype)
                - dense_loss(minus, device=device, dtype=dtype)
            )
            / (2 * epsilon)
        )
        reverse_direction = sum(
            gradient * delta for gradient, delta in zip(first_gradients, direction)
        )
        directional_error = abs(finite_difference - reverse_direction)
        max_value = max(item["value_error"] for item in step_records)
        max_gradient = max(max(item["gradient_errors"]) for item in step_records)
        max_parameter = max(max(item["parameter_errors"]) for item in step_records)
        approximate_reference_parameters = tuple(
            value.detach().clone().requires_grad_(True) for value in parameters
        )
        approximate_reference = run_mps(
            workload(approximate_reference_parameters, device=device, dtype=dtype),
            max_bond=2,
        )
        approximate_reference_value = approximate_reference.expectation_ps(
            z=(1, 6)
        ).mean()
        approximate_reference_value.backward()
        approximate_value_error = abs(
            float(approximate.value) - float(approximate_reference_value.detach())
        )
        approximate_gradient_error = max(
            abs(float(actual.grad) - float(reference.grad))
            for actual, reference in zip(
                approximate_parameters, approximate_reference_parameters
            )
        )
        rank_errors = []
        for rank_index in range(world):
            rank_errors.append(
                {
                    "rank": rank_index,
                    "max_value_error": max(
                        abs(
                            item["rank_observations"][rank_index]["value"]
                            - item["dense_value"]
                        )
                        for item in step_records
                    ),
                    "max_parameter_gradient_error": max(
                        abs(observed - reference)
                        for item in step_records
                        for observed, reference in zip(
                            item["rank_observations"][rank_index]["gradients"],
                            item["dense_gradients"],
                        )
                    ),
                }
            )
        workload_payload = {
            "seed": SEED,
            "n_wires": 8,
            "steps": args.steps,
            "dtype": args.dtype,
            "optimizer": args.optimizer,
        }
        case = {
            "schema": "flagquantum.issue091.mps_correctness_case.v1",
            "seed": SEED,
            "world_size": world,
            "backend": args.backend,
            "dtype": args.dtype,
            "optimizer": args.optimizer,
            "steps": args.steps,
            "n_wires": 8,
            "boundary_edges": [
                [wire, wire + 1]
                for wire in range(7)
                if (wire * world // 8) != ((wire + 1) * world // 8)
            ],
            "gradient_ownership": "deterministic_all_reduce",
            "max_value_error": max_value,
            "max_gradient_error": max_gradient,
            "max_parameter_error": max_parameter,
            "boundary_directional_derivative_error": directional_error,
            "boundary_directional_derivative_passed": directional_error
            <= TOLERANCES["directional_atol"],
            "exact_discarded_weight": exact_discarded,
            "approximate_discarded_weight": float(approximate.discarded_weight),
            "approximate_error_budget": approximate_budget,
            "approximate_value_error": approximate_value_error,
            "approximate_gradient_error": approximate_gradient_error,
            "rank_errors": rank_errors,
            "workload_sha256": hashlib.sha256(
                json.dumps(workload_payload, sort_keys=True).encode()
            ).hexdigest(),
            "commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "command": " ".join(sys.argv),
            "step_records": step_records,
            "torch_version": torch.__version__,
            "platform": platform.platform(),
        }
        case["passed"] = (
            max_value <= TOLERANCES["value_atol"]
            and max_gradient <= TOLERANCES["gradient_atol"]
            and max_parameter <= TOLERANCES["parameter_atol"]
            and directional_error <= TOLERANCES["directional_atol"]
            and exact_discarded == 0.0
            and float(approximate.discarded_weight) <= approximate_budget
            and approximate_value_error <= TOLERANCES["approximate_value_atol"]
            and approximate_gradient_error <= TOLERANCES["approximate_gradient_atol"]
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(case, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"output": str(args.output), "passed": case["passed"]}))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
