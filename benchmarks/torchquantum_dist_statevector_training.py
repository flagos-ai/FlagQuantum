#!/usr/bin/env python3
"""Matched TorchQuantum-Dist invertible value-and-gradient benchmark.

This script intentionally imports IonQ's unmodified TQD source from an isolated
benchmark checkout.  It is not a FlagQuantum runtime dependency.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

import tqd
import tqd.module

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.sc27_metadata import (  # noqa: E402
    build_optimizer,
    driver_version,
    gpu_identity,
    optimizer_state_bytes,
    parameter_vector,
    reference_errors,
    source_identity,
    tensor_bytes,
    topology_snapshot,
)

SCHEMA = "flagquantum.external.torchquantum_dist_training.v1"


def parameters(n_wires: int, layers: int, seed: int, device: torch.device) -> torch.Tensor:
    return torch.tensor(
        [
            0.11 + 0.013 * ((seed + layer * n_wires + wire) % 37)
            for layer in range(layers)
            for wire in range(n_wires)
        ],
        dtype=torch.float32,
        device=device,
    ).reshape(1, -1).requires_grad_()


def circuit_modules(n_wires: int, layers: int) -> tqd.module.InvertibleUnitary:
    gates: list[torch.nn.Module] = []
    cursor = 0
    for layer in range(layers):
        functions = []
        for wire in range(n_wires):
            functions.append({"func": "ry", "wires": [wire], "input_idx": [cursor]})
            cursor += 1
        gates.append(tqd.GeneralEncoder(functions))
        edges = (
            [(wire, wire + 1) for wire in range(n_wires - 1)]
            if layer % 2 == 0
            else [(wire + 1, wire) for wire in range(n_wires - 2, -1, -1)]
        )
        gates.extend(tqd.CX(wires=[control, target]) for control, target in edges)
    module = tqd.module.InvertibleUnitary(gates)
    module.train()
    return module


def timing(samples: list[float]) -> dict[str, Any]:
    mean = statistics.fmean(samples)
    return {
        "samples_seconds": samples,
        "sample_count": len(samples),
        "median_seconds": statistics.median(samples),
        "mean_seconds": mean,
        "coefficient_of_variation": statistics.pstdev(samples) / mean,
    }


def max_across_ranks(value: float) -> float:
    tensor = torch.tensor(value, dtype=torch.float64, device="cuda")
    if dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return float(tensor.cpu())


def run(args: argparse.Namespace) -> dict[str, Any] | None:
    if args.n_wires < 2 or args.layers < 1:
        raise ValueError("n_wires >= 2 and layers >= 1 required")
    if args.warmup < 0 or args.repetitions < 1:
        raise ValueError("warmup >= 0 and repetitions >= 1 required")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size & (world_size - 1):
        raise ValueError("TorchQuantum-Dist requires a power-of-two world size")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    qdev = tqd.DistributedQuantumDevice(
        args.n_wires,
        bsz=1,
        device="cuda",
        world_sz=world_size,
        invertible=True,
    )
    module = circuit_modules(args.n_wires, args.layers)
    values = parameters(args.n_wires, args.layers, args.seed, device)
    optimizer = build_optimizer(
        args.optimizer, [values], learning_rate=args.learning_rate
    )

    def value_and_grad() -> tuple[
        float, tuple[float, ...], float, float, float, int
    ]:
        if optimizer is None:
            values.grad = None
        else:
            optimizer.zero_grad(set_to_none=True)
        qdev.reset_states()
        if dist.is_initialized():
            dist.barrier()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        module(qdev, values)
        result = tqd.measure_allZ(qdev, shots=0, training=True)[
            0, args.n_wires // 2
        ]
        torch.cuda.synchronize(device)
        forward_seconds = time.perf_counter() - started
        started = time.perf_counter()
        result.backward()
        # TQD documents parameter gradients as Partial DTensors.  Its public
        # example omits optimizer synchronization; sum them here so this is an
        # optimizer-ready distributed training step.
        if dist.is_initialized():
            dist.all_reduce(values.grad, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize(device)
        backward_seconds = time.perf_counter() - started
        started = time.perf_counter()
        if optimizer is not None:
            optimizer.step()
        torch.cuda.synchronize(device)
        optimizer_seconds = time.perf_counter() - started
        peak_bytes = torch.cuda.max_memory_allocated(device)
        forward_seconds = max_across_ranks(forward_seconds)
        backward_seconds = max_across_ranks(backward_seconds)
        optimizer_seconds = max_across_ranks(optimizer_seconds)
        peak_bytes = int(max_across_ranks(float(peak_bytes)))
        return (
            float(result.detach().cpu()),
            tuple(float(item) for item in values.grad.detach().flatten().cpu()),
            forward_seconds,
            backward_seconds,
            optimizer_seconds,
            peak_bytes,
        )

    initial_value: float | None = None
    initial_gradients: tuple[float, ...] | None = None
    for index in range(args.warmup):
        value, gradients, _, _, _, _ = value_and_grad()
        if initial_value is None:
            initial_value, initial_gradients = value, gradients
        if rank == 0:
            print(f"heartbeat phase=warmup step={index + 1}/{args.warmup}", flush=True)
    forward_samples: list[float] = []
    backward_samples: list[float] = []
    optimizer_samples: list[float] = []
    peak_samples: list[int] = []
    local_peak_samples: list[int] = []
    for index in range(args.repetitions):
        (
            value,
            gradients,
            forward_seconds,
            backward_seconds,
            optimizer_seconds,
            peak_bytes,
        ) = value_and_grad()
        initial_value = value if initial_value is None else initial_value
        initial_gradients = gradients if initial_gradients is None else initial_gradients
        forward_samples.append(forward_seconds)
        backward_samples.append(backward_seconds)
        optimizer_samples.append(optimizer_seconds)
        peak_samples.append(peak_bytes)
        local_peak_samples.append(int(torch.cuda.max_memory_allocated(device)))
        if rank == 0:
            print(
                f"heartbeat phase=measurement step={index + 1}/{args.repetitions} "
                f"forward={forward_seconds:.6f}s backward={backward_seconds:.6f}s "
                f"optimizer={optimizer_seconds:.6f}s",
                flush=True,
            )
    assert initial_value is not None and initial_gradients is not None
    placement = {
        "rank": rank,
        "local_rank": local_rank,
        "hostname": platform.node(),
        "device": str(device),
        **gpu_identity(local_rank),
        "topology": topology_snapshot(),
    }
    placements: list[dict[str, Any]] = [placement]
    local_peak = max(local_peak_samples)
    peak_bytes_by_rank = [local_peak]
    parameter_bytes_by_rank = [tensor_bytes([values])]
    gradient_bytes_by_rank = [tensor_bytes([values.grad])]
    optimizer_bytes_by_rank = [optimizer_state_bytes(optimizer)]
    parameter_vectors = [parameter_vector([values]).tolist()]
    if dist.is_initialized():
        placements = [None] * world_size  # type: ignore[list-item]
        dist.all_gather_object(placements, placement)
        for local_value, target in (
            (local_peak, peak_bytes_by_rank),
            (tensor_bytes([values]), parameter_bytes_by_rank),
            (tensor_bytes([values.grad]), gradient_bytes_by_rank),
            (optimizer_state_bytes(optimizer), optimizer_bytes_by_rank),
        ):
            gathered: list[int | None] = [None] * world_size
            dist.all_gather_object(gathered, local_value)
            target[:] = [int(item) for item in gathered if item is not None]
        vectors: list[list[float] | None] = [None] * world_size
        dist.all_gather_object(vectors, parameter_vector([values]).tolist())
        parameter_vectors = [item for item in vectors if item is not None]
    parameters_equal = all(vector == parameter_vectors[0] for vector in parameter_vectors)
    workload = {
        "name": "full_width_linear_hea",
        "n_wires": args.n_wires,
        "layers": args.layers,
        "parameter_count": args.n_wires * args.layers,
        "observable": f"Z({args.n_wires // 2})",
        "dtype": "complex64",
        "seed": args.seed,
        "world_size": world_size,
    }
    payload = {
        "schema": SCHEMA,
        "benchmark": "torchquantum_dist_invertible_training",
        "artifact_class": "measured_external_development_run",
        "claim_evidence_type": "external_value_and_gradient_comparison",
        "comparison_class": "matched_differentiable_execution",
        "release_gate_allowed": False,
        "scalability_claim_allowed": False,
        "world_size": world_size,
        "rank_placement": placements,
        "source_identity": source_identity(
            repo_root=REPO_ROOT,
            workload=workload,
            container_digest=args.container_digest,
            raw_log_sha256=args.raw_log_sha256,
        ),
        "distribution_semantics": (
            "sharded_across_dtensor_ranks"
            if world_size > 1
            else "single_device_fast_path"
        ),
        "external_baseline_isolation": {
            "purpose": "benchmark_only",
            "flagquantum_runtime_dependency": False,
            "provider": "IonQ TorchQuantum-Dist",
            "source_import": "unmodified_official_checkout",
            "gradient_path": "InvertibleUnitary plus optimizer-ready gradient sum",
        },
        "environment": {
            "hostname": platform.node(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device_name": torch.cuda.get_device_name(local_rank),
            "rank": rank,
            "local_rank": local_rank,
            "world_size": world_size,
            "driver_version": driver_version(),
        },
        "workload": {
            "name": "full_width_linear_hea",
            "n_wires": args.n_wires,
            "layers": args.layers,
            "gate_count": (2 * args.n_wires - 1) * args.layers,
            "parameter_count": args.n_wires * args.layers,
            "gate_set": ["RY", "CNOT"],
            "entanglement": "directed_cnot_linear",
            "observable": f"Z({args.n_wires // 2})",
            "measurement_implementation": "TQD measure_allZ then select target",
            "dtype": "complex64",
            "storage_representation": "float32_real_imag",
            "parameter_dtype": "float32",
            "seed": args.seed,
        },
        "protocol": {
            "interface": "torch",
            "gradient_method": "invertible_reverse_mode",
            "fixed_parameters_across_samples": optimizer is None,
            "warmup": args.warmup,
            "repetitions": args.repetitions,
            "independent_run_index": args.independent_run_index,
            "synchronization": "cuda_device_synchronized_wall_clock",
            "distributed_sample_reduction": "rank_max",
            "parameter_gradient_reduction": "sum_after_backward",
        },
        "correctness": {
            "initial_value": initial_value,
            "initial_gradients": initial_gradients,
            "values_and_gradients_finite": bool(
                torch.isfinite(torch.tensor([initial_value, *initial_gradients])).all()
            ),
            **reference_errors(
                value=initial_value,
                gradients=initial_gradients,
                reference_path=args.correctness_reference,
            ),
        },
        "forward": timing(forward_samples),
        "backward": timing(backward_samples),
        "optimizer": {
            "name": args.optimizer,
            "executed": optimizer is not None,
            "step_seconds": timing(optimizer_samples),
            "parameters_equal_across_ranks": parameters_equal,
        },
        "value_and_grad": timing(
            [fwd + bwd for fwd, bwd in zip(forward_samples, backward_samples)]
        ),
        "training_step": timing(
            [
                fwd + bwd + opt
                for fwd, bwd, opt in zip(
                    forward_samples, backward_samples, optimizer_samples
                )
            ]
        ),
        "ownership": {
            "primal_state": (
                "sharded_across_dtensor_ranks" if world_size > 1 else "single_device"
            ),
            "adjoint_state": (
                "invertible_rematerialized_sharded_across_dtensor_ranks"
                if world_size > 1
                else "single_device"
            ),
            "parameter_gradient": "replicated_after_all_reduce",
            "optimizer_state": "replicated_across_ranks",
            "parameter_bytes_per_rank": parameter_bytes_by_rank,
            "gradient_bytes_per_rank": gradient_bytes_by_rank,
            "optimizer_bytes_per_rank": optimizer_bytes_by_rank,
        },
        "memory": {
            "measurement": "torch.cuda.max_memory_allocated",
            "peak_bytes_per_rank_max": max(peak_samples),
            "samples_per_rank_max_bytes": peak_samples,
            "peak_bytes_by_rank": peak_bytes_by_rank,
        },
        "communication": {
            "bytes_by_rank": None,
            "measurement": (
                "TQD internal DTensor redistribution bytes unavailable from public API; "
                "the explicit parameter-gradient all-reduce is one parameter vector per rank"
            ),
            "known_gradient_all_reduce_payload_bytes_per_rank": tensor_bytes([values.grad]),
        },
        "fallback_events": [],
    }
    if dist.is_initialized():
        dist.barrier()
    return payload if rank == 0 else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-wires", type=int, required=True)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--optimizer", choices=("none", "sgd", "adam"), default="adam")
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--independent-run-index", type=int, choices=(1, 2, 3))
    parser.add_argument("--container-digest")
    parser.add_argument("--raw-log-sha256")
    parser.add_argument("--correctness-reference", type=Path)
    parser.add_argument("--json-output", type=Path, required=True)
    args = parser.parse_args()
    payload = run(args)
    if payload is not None:
        encoded = json.dumps(payload, indent=2, sort_keys=True)
        print(encoded)
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(encoded + "\n", encoding="utf-8")
        if not payload["correctness"]["values_and_gradients_finite"]:
            raise SystemExit(2)
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
