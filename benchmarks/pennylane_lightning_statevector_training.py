#!/usr/bin/env python3
"""Matched PennyLane Lightning-GPU adjoint value-and-gradient benchmark.

This external baseline runs only in an isolated benchmark environment.
PennyLane and cuQuantum are never FlagQuantum runtime dependencies.
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

import numpy as np
import pennylane as qml
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.gpu_memory_sampler import NvmlMemorySampler  # noqa: E402

SCHEMA = "flagquantum.external.pennylane_lightning_gpu_training.v1"


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

    def value_and_grad() -> tuple[
        float, tuple[float, ...], float, float, int, int
    ]:
        values.grad = None
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
        peak_per_rank = memory.peak_bytes
        peak_aggregate = memory.peak_bytes
        if comm is not None:
            forward_seconds = comm.allreduce(forward_seconds, op=MPI.MAX)
            backward_seconds = comm.allreduce(backward_seconds, op=MPI.MAX)
            peak_per_rank = comm.allreduce(peak_per_rank, op=MPI.MAX)
            peak_aggregate = comm.allreduce(peak_aggregate, op=MPI.SUM)
        assert values.grad is not None
        return (
            float(result.detach()),
            tuple(float(item) for item in values.grad.detach()),
            forward_seconds,
            backward_seconds,
            peak_per_rank,
            peak_aggregate,
        )

    initial_value = None
    initial_gradients = None
    for _ in range(args.warmup):
        initial_value, initial_gradients, _, _, _, _ = value_and_grad()
    forward_samples: list[float] = []
    backward_samples: list[float] = []
    value_and_grad_samples: list[float] = []
    peak_memory_per_rank_samples: list[int] = []
    peak_memory_aggregate_samples: list[int] = []
    for _ in range(args.repetitions):
        (
            value,
            gradients,
            forward_seconds,
            backward_seconds,
            peak_per_rank,
            peak_aggregate,
        ) = value_and_grad()
        if initial_value is None:
            initial_value, initial_gradients = value, gradients
        forward_samples.append(forward_seconds)
        backward_samples.append(backward_seconds)
        value_and_grad_samples.append(forward_seconds + backward_seconds)
        peak_memory_per_rank_samples.append(peak_per_rank)
        peak_memory_aggregate_samples.append(peak_aggregate)
    assert initial_value is not None and initial_gradients is not None
    return {
        "schema": SCHEMA,
        "benchmark": "pennylane_lightning_gpu_adjoint_training",
        "artifact_class": "measured_external_development_run",
        "claim_evidence_type": "external_value_and_gradient_comparison",
        "release_gate_allowed": False,
        "scalability_claim_allowed": False,
        "distribution_semantics": (
            "sharded_across_mpi_ranks" if args.mpi else "single_device_fast_path"
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
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device_name": torch.cuda.get_device_name(local_rank),
            "rank": rank,
            "local_rank": local_rank,
            "world_size": world_size,
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
            "dtype": "complex64",
            "parameter_dtype": "float32",
            "seed": args.seed,
        },
        "protocol": {
            "interface": "torch",
            "gradient_method": "adjoint",
            "fixed_parameters_across_samples": True,
            "warmup": args.warmup,
            "repetitions": args.repetitions,
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
        },
        "forward": _timing(forward_samples),
        "backward": _timing(backward_samples),
        "value_and_grad": _timing(value_and_grad_samples),
        "memory": {
            "measurement": "NVML device-used bytes sampled every 20 ms",
            "exclusive_gpu_required": True,
            "peak_bytes_per_rank_max": max(peak_memory_per_rank_samples),
            "peak_bytes_aggregate_max": max(peak_memory_aggregate_samples),
            "samples_per_rank_max_bytes": peak_memory_per_rank_samples,
            "samples_aggregate_bytes": peak_memory_aggregate_samples,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-wires", type=int, required=True)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=440044)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=30)
    parser.add_argument("--mpi", action="store_true")
    parser.add_argument("--mpi-buf-size", type=int, default=64)
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
