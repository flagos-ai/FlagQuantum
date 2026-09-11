#!/usr/bin/env python3
"""Matched PennyLane Lightning-GPU adjoint value-and-gradient benchmark.

This external baseline runs only in an isolated benchmark environment.
PennyLane and cuQuantum are never FlagQuantum runtime dependencies.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pennylane as qml
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.gpu_memory_sampler import NvmlMemorySampler  # noqa: E402
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

SCHEMA = "flagquantum.external.pennylane_lightning_gpu_training.v1"


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def parameters(n_wires: int, layers: int, seed: int) -> torch.Tensor:
    return torch.tensor(
        [
            0.11 + 0.013 * ((seed + layer * n_wires + wire) % 37)
            for layer in range(layers)
            for wire in range(n_wires)
        ],
        dtype=torch.float32,
        requires_grad=True,
    )


def build_qnode(
    n_wires: int, layers: int, *, mpi: bool, mpi_buf_size: int
) -> Any:
    options: dict[str, Any] = {
        "wires": n_wires,
        "c_dtype": np.complex64,
    }
    if mpi:
        options.update({"mpi": True, "mpi_buf_size": mpi_buf_size})
    device = qml.device("lightning.gpu", **options)

    @qml.qnode(device, interface="torch", diff_method="adjoint")
    def circuit(values: torch.Tensor) -> Any:
        cursor = 0
        for layer in range(layers):
            for wire in range(n_wires):
                qml.RY(values[cursor], wires=wire)
                cursor += 1
            edges = (
                [(wire, wire + 1) for wire in range(n_wires - 1)]
                if layer % 2 == 0
                else [(wire + 1, wire) for wire in range(n_wires - 2, -1, -1)]
            )
            for control, target in edges:
                qml.CNOT(wires=(control, target))
        return qml.expval(qml.PauliZ(n_wires // 2))

    return circuit


def _timing(samples: list[float]) -> dict[str, Any]:
    mean = statistics.fmean(samples)
    return {
        "samples_seconds": samples,
        "sample_count": len(samples),
        "median_seconds": statistics.median(samples),
        "mean_seconds": mean,
        "coefficient_of_variation": statistics.pstdev(samples) / mean,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.n_wires < 2 or args.layers < 1:
        raise ValueError("n_wires >= 2 and layers >= 1 required")
    if args.warmup < 0 or args.repetitions < 3:
        raise ValueError("warmup >= 0 and repetitions >= 3 required")
    comm = None
    rank = 0
    world_size = 1
    local_rank = 0
    if args.mpi:
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        world_size = comm.Get_size()
        local_rank = int(os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    qnode = build_qnode(
        args.n_wires,
        args.layers,
        mpi=args.mpi,
        mpi_buf_size=args.mpi_buf_size,
    )
    values = parameters(args.n_wires, args.layers, args.seed)
    optimizer = build_optimizer(
        args.optimizer, [values], learning_rate=args.learning_rate
    )

    def value_and_grad() -> tuple[
        float, tuple[float, ...], float, float, float, int, int, int
    ]:
        if optimizer is None:
            values.grad = None
        else:
            optimizer.zero_grad(set_to_none=True)
        if comm is not None:
            comm.Barrier()
        with NvmlMemorySampler(local_rank) as memory:
            torch.cuda.synchronize()
            forward_started = time.perf_counter()
            result = qnode(values)
            torch.cuda.synchronize()
            forward_seconds = time.perf_counter() - forward_started
            backward_started = time.perf_counter()
            result.backward()
            torch.cuda.synchronize()
            backward_seconds = time.perf_counter() - backward_started
            optimizer_started = time.perf_counter()
            if optimizer is not None:
                optimizer.step()
            torch.cuda.synchronize()
            optimizer_seconds = time.perf_counter() - optimizer_started
        local_peak_bytes = int(memory.peak_bytes)
        peak_per_rank = local_peak_bytes
        peak_aggregate = local_peak_bytes
        if comm is not None:
            forward_seconds = comm.allreduce(forward_seconds, op=MPI.MAX)
            backward_seconds = comm.allreduce(backward_seconds, op=MPI.MAX)
            optimizer_seconds = comm.allreduce(optimizer_seconds, op=MPI.MAX)
            peak_per_rank = comm.allreduce(peak_per_rank, op=MPI.MAX)
            peak_aggregate = comm.allreduce(peak_aggregate, op=MPI.SUM)
        assert values.grad is not None
        return (
            float(result.detach()),
            tuple(float(item) for item in values.grad.detach()),
            forward_seconds,
            backward_seconds,
            optimizer_seconds,
            peak_per_rank,
            peak_aggregate,
            local_peak_bytes,
        )

    initial_value = None
    initial_gradients = None
    for _ in range(args.warmup):
        value, gradients, _, _, _, _, _, _ = value_and_grad()
        if initial_value is None:
            initial_value, initial_gradients = value, gradients
    forward_samples: list[float] = []
    backward_samples: list[float] = []
    optimizer_samples: list[float] = []
    value_and_grad_samples: list[float] = []
    training_step_samples: list[float] = []
    peak_memory_per_rank_samples: list[int] = []
    peak_memory_aggregate_samples: list[int] = []
    local_peak_memory_samples: list[int] = []
    for _ in range(args.repetitions):
        (
            value,
            gradients,
            forward_seconds,
            backward_seconds,
            optimizer_seconds,
            peak_per_rank,
            peak_aggregate,
            local_peak_bytes,
        ) = value_and_grad()
        if initial_value is None:
            initial_value, initial_gradients = value, gradients
        forward_samples.append(forward_seconds)
        backward_samples.append(backward_seconds)
        optimizer_samples.append(optimizer_seconds)
        value_and_grad_samples.append(forward_seconds + backward_seconds)
        training_step_samples.append(
            forward_seconds + backward_seconds + optimizer_seconds
        )
        peak_memory_per_rank_samples.append(peak_per_rank)
        peak_memory_aggregate_samples.append(peak_aggregate)
        local_peak_memory_samples.append(local_peak_bytes)
    assert initial_value is not None and initial_gradients is not None
    placement = {
        "rank": rank,
        "local_rank": local_rank,
        "hostname": platform.node(),
        **gpu_identity(local_rank),
        "topology": topology_snapshot(),
    }
    placements = [placement]
    peak_bytes_by_rank = [max(local_peak_memory_samples)]
    parameter_vectors = [parameter_vector([values]).tolist()]
    local_parameter_bytes = tensor_bytes([values])
    local_gradient_bytes = tensor_bytes([values.grad]) if values.grad is not None else 0
    measured_optimizer_bytes = optimizer_state_bytes(optimizer)
    parameter_bytes_by_rank = [local_parameter_bytes]
    gradient_bytes_by_rank = [local_gradient_bytes]
    optimizer_bytes_by_rank = [measured_optimizer_bytes]
    if comm is not None:
        placements = comm.allgather(placement)
        peak_bytes_by_rank = comm.allgather(max(local_peak_memory_samples))
        parameter_vectors = comm.allgather(parameter_vector([values]).tolist())
        parameter_bytes_by_rank = comm.allgather(local_parameter_bytes)
        gradient_bytes_by_rank = comm.allgather(local_gradient_bytes)
        optimizer_bytes_by_rank = comm.allgather(measured_optimizer_bytes)
    parameters_equal_across_ranks = all(
        item == parameter_vectors[0] for item in parameter_vectors
    )
    distributed = args.mpi and world_size > 1
    workload = {
        "name": "full_width_linear_hea",
        "n_wires": args.n_wires,
        "layers": args.layers,
        "gate_count": (2 * args.n_wires - 1) * args.layers,
        "parameter_count": args.n_wires * args.layers,
        "gate_set": ["RY", "CNOT"],
        "entanglement": "directed_cnot_linear",
        "observable": f"Z({args.n_wires // 2})",
        "dtype": "complex64",
        "parameter_dtype": "float32",
        "seed": args.seed,
    }
    return {
        "schema": SCHEMA,
        "benchmark": "pennylane_lightning_gpu_adjoint_training",
        "artifact_class": "measured_external_development_run",
        "claim_evidence_type": "external_value_and_gradient_comparison",
        "release_gate_allowed": False,
        "scalability_claim_allowed": False,
        "distribution_semantics": (
            "sharded_across_mpi_ranks" if distributed else "single_device_fast_path"
        ),
        "world_size": world_size,
        "local_world_size": int(
            os.environ.get("OMPI_COMM_WORLD_LOCAL_SIZE", world_size)
        ),
        "node_count": len({item["hostname"] for item in placements}),
        "rank_placement": placements,
        "source_identity": source_identity(
            repo_root=REPO_ROOT,
            workload=workload,
            container_digest=args.container_digest,
            raw_log_sha256=args.raw_log_sha256,
        ),
        "external_baseline_isolation": {
            "purpose": "benchmark_only",
            "flagquantum_runtime_dependency": False,
            "provider": "PennyLane Lightning-GPU",
            "cuquantum_backed": True,
        },
        "environment": {
            "hostname": platform.node(),
            "python": platform.python_version(),
            "pennylane": qml.__version__,
            "pennylane_lightning": _package_version("pennylane-lightning"),
            "cuquantum_python": _package_version("cuquantum-python"),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device_name": torch.cuda.get_device_name(local_rank),
            "rank": rank,
            "local_rank": local_rank,
            "world_size": world_size,
            "driver_version": driver_version(),
        },
        "workload": workload,
        "protocol": {
            "interface": "torch",
            "gradient_method": "adjoint",
            "fixed_parameters_across_samples": optimizer is None,
            "optimizer_included": optimizer is not None,
            "optimizer": args.optimizer,
            "learning_rate": args.learning_rate,
            "warmup": args.warmup,
            "repetitions": args.repetitions,
            "independent_run_index": args.independent_run_index,
            "synchronization": "cuda_device_synchronized_wall_clock",
            "mpi": args.mpi,
            "mpi_buf_size_mib": args.mpi_buf_size if args.mpi else None,
            "distributed_sample_reduction": (
                "rank_max" if args.mpi else "not_applicable"
            ),
        },
        "correctness": {
            "initial_value": initial_value,
            "initial_gradients": initial_gradients,
            "values_and_gradients_finite": bool(
                np.isfinite(initial_value)
                and np.isfinite(np.asarray(initial_gradients)).all()
            ),
            **reference_errors(
                value=initial_value,
                gradients=initial_gradients,
                reference_path=args.correctness_reference,
            ),
        },
        "forward": _timing(forward_samples),
        "backward": _timing(backward_samples),
        "optimizer_timing": _timing(optimizer_samples),
        "value_and_grad": _timing(value_and_grad_samples),
        "training_step": _timing(training_step_samples),
        "memory": {
            "measurement": "NVML device-used bytes sampled every 20 ms",
            "exclusive_gpu_required": True,
            "peak_bytes_per_rank_max": max(peak_memory_per_rank_samples),
            "peak_bytes_aggregate_max": max(peak_memory_aggregate_samples),
            "samples_per_rank_max_bytes": peak_memory_per_rank_samples,
            "samples_aggregate_bytes": peak_memory_aggregate_samples,
            "peak_bytes_by_rank": peak_bytes_by_rank,
        },
        "communication": {
            "bytes_by_rank": None,
            "measurement": "not_exposed_by_lightning_gpu_python_interface",
        },
        "ownership": {
            "primal_state": (
                "backend_documented_sharded_across_ranks"
                if distributed
                else "single_device"
            ),
            "adjoint_state": (
                "backend_documented_sharded_across_ranks"
                if distributed
                else "single_device"
            ),
            "parameter_gradient": (
                "replicated_across_ranks" if distributed else "single_device"
            ),
            "optimizer_state": (
                "replicated_across_ranks"
                if distributed and optimizer is not None
                else "not_present" if optimizer is None else "single_device"
            ),
            "full_quantum_state_materialized": "not_observable_from_python_api",
            "parameter_bytes_per_rank": parameter_bytes_by_rank,
            "gradient_bytes_per_rank": gradient_bytes_by_rank,
            "optimizer_bytes_per_rank": optimizer_bytes_by_rank,
        },
        "optimizer": {
            "name": args.optimizer,
            "executed": optimizer is not None,
            "step_seconds": statistics.median(optimizer_samples),
            "parameters_equal_across_ranks": parameters_equal_across_ranks,
            "state_bytes_per_rank": measured_optimizer_bytes,
        },
        "fallback_events": [],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-wires", type=int, required=True)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--mpi", action="store_true")
    parser.add_argument("--mpi-buf-size", type=int, default=64)
    parser.add_argument(
        "--optimizer", choices=("none", "sgd", "adam"), default="adam"
    )
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--independent-run-index", type=int, choices=(1, 2, 3))
    parser.add_argument("--container-digest")
    parser.add_argument("--raw-log-sha256")
    parser.add_argument("--correctness-reference", type=Path)
    parser.add_argument("--json-output", type=Path, required=True)
    args = parser.parse_args()
    payload = run(args)
    if args.mpi:
        from mpi4py import MPI

        if MPI.COMM_WORLD.Get_rank() != 0:
            return
    encoded = json.dumps(payload, indent=2, sort_keys=True)
    print(encoded)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(encoded + "\n", encoding="utf-8")
    if not payload["correctness"]["values_and_gradients_finite"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
