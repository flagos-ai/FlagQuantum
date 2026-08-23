"""Measured multi-GPU sparse-output tensor-network capacity case."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import socket
import subprocess
import time
from pathlib import Path

import torch
import torch.distributed as dist

import flagquantum as fq
from flagquantum.compilation import load_tn_working_set_calibration
from flagquantum.runtime.distributed.tensor_network_execution import (
    _persistent_plan_key,
)
from flagquantum.simulation.tensor_contraction import (
    _build_slicing_plan,
    _cost_for_sliced_labels,
    _parallel_slice_labels,
)
from flagquantum.simulation.tensor_execution import _amplitude_batch_projection


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qubits", type=int, required=True)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--targets", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument(
        "--dtype", choices=("complex64", "complex128"), default="complex64"
    )
    parser.add_argument("--grid-rows", type=int)
    parser.add_argument("--grid-cols", type=int)
    parser.add_argument("--max-intermediate-gib", type=float)
    parser.add_argument("--max-working-set-gib", type=float)
    parser.add_argument("--working-set-safety-factor", type=float, default=4.0)
    parser.add_argument("--plan-cache-path", type=Path)
    parser.add_argument("--memory-calibration", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _circuit(
    n_qubits: int, depth: int, *, device: str, dtype: torch.dtype
) -> fq.Circuit:
    circuit = fq.Circuit(n_qubits, device=device, dtype=dtype)
    for layer in range(depth):
        for wire in range(n_qubits):
            circuit.ry(wire, theta=0.007 * (1 + layer + wire % 7))
        start = layer % 2
        for wire in range(start, n_qubits - 1, 2):
            circuit.cx(wire, wire + 1)
    return circuit


def _grid_circuit(
    rows: int,
    columns: int,
    cycles: int,
    *,
    device: str,
    dtype: torch.dtype,
) -> fq.Circuit:
    circuit = fq.Circuit(rows * columns, device=device, dtype=dtype)
    for cycle in range(cycles):
        for wire in range(rows * columns):
            circuit.ry(wire, theta=0.01 * (cycle + 1))
        for row in range(rows):
            for column in range(columns - 1):
                left = row * columns + column
                circuit.cx(left, left + 1)
        for row in range(rows - 1):
            for column in range(columns):
                top = row * columns + column
                circuit.cx(top, top + columns)
    return circuit


def _targets(n_qubits: int, count: int) -> tuple[str, ...]:
    modulus = 1 << n_qubits
    values = [0]
    for index in range(1, count):
        values.append((index * 0x9E3779B97F4A7C15) % modulus)
    return tuple(f"{value:0{n_qubits}b}" for value in values)


def _git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def main() -> None:
    args = _parser().parse_args()
    memory_calibration = (
        None
        if args.memory_calibration is None
        else load_tn_working_set_calibration(args.memory_calibration)
    )
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    device = f"cuda:{local_rank}"
    dtype = getattr(torch, args.dtype)
    dist.init_process_group("nccl")

    phase_start = time.perf_counter()
    grid_enabled = args.grid_rows is not None or args.grid_cols is not None
    if grid_enabled and (args.grid_rows is None or args.grid_cols is None):
        raise ValueError("--grid-rows and --grid-cols must be supplied together")
    n_qubits = (
        int(args.grid_rows) * int(args.grid_cols) if grid_enabled else args.qubits
    )
    circuit = (
        _grid_circuit(
            int(args.grid_rows),
            int(args.grid_cols),
            args.depth,
            device=device,
            dtype=dtype,
        )
        if grid_enabled
        else _circuit(n_qubits, args.depth, device=device, dtype=dtype)
    )
    circuit_build_seconds = time.perf_counter() - phase_start
    targets = _targets(n_qubits, args.targets)
    phase_start = time.perf_counter()
    probe = fq.build_tensor_network(circuit, device=device, dtype=dtype)
    tn_build_seconds = time.perf_counter() - phase_start
    phase_start = time.perf_counter()
    nodes, outputs = _amplitude_batch_projection(probe, targets)
    plan_payload = [None]
    if rank == 0:
        effective_intermediate_gib = args.max_intermediate_gib
        if args.max_working_set_gib is not None:
            if memory_calibration is None:
                workset_intermediate_gib = (
                    args.max_working_set_gib / args.working_set_safety_factor
                )
            else:
                available_bytes = (
                    int(args.max_working_set_gib * 1024**3)
                    - memory_calibration.fixed_reserved_overhead_bytes
                )
                workset_intermediate_gib = (
                    available_bytes
                    / memory_calibration.recommended_safety_factor
                    / 1024**3
                )
            effective_intermediate_gib = (
                workset_intermediate_gib
                if effective_intermediate_gib is None
                else min(effective_intermediate_gib, workset_intermediate_gib)
            )
        effective_intermediate_bytes = (
            None
            if effective_intermediate_gib is None
            else int(effective_intermediate_gib * 1024**3)
        )
        preflight_key = _persistent_plan_key(
            nodes,
            outputs,
            max_intermediate_size=None,
            max_intermediate_bytes=effective_intermediate_bytes,
            sliced_labels=None,
        )
        preflight_path = (
            None
            if args.plan_cache_path is None
            else args.plan_cache_path.with_suffix(
                args.plan_cache_path.suffix + ".preflight.json"
            )
        )
        cached_preflight = None
        if preflight_path is not None and preflight_path.is_file():
            candidate = json.loads(preflight_path.read_text(encoding="utf-8"))
            if candidate.get("cache_key") == preflight_key:
                cached_preflight = candidate
        if cached_preflight is not None:
            sliced_labels = tuple(cached_preflight["sliced_labels"])
            expected_slice_tasks = int(cached_preflight["expected_slice_tasks"])
            slice_cost = cached_preflight["slice_cost"]
        else:
            if effective_intermediate_bytes is not None:
                slicing = _build_slicing_plan(
                    nodes,
                    outputs,
                    max_intermediate_bytes=effective_intermediate_bytes,
                )
                sliced_labels = slicing.sliced_labels
                expected_slice_tasks = slicing.n_slices
            else:
                slice_axes = (
                    int(math.ceil(math.log2(world_size))) if world_size > 1 else 0
                )
                sliced_labels = _parallel_slice_labels(nodes, outputs, world_size)
                expected_slice_tasks = 1 << slice_axes
            slice_cost = _cost_for_sliced_labels(
                nodes,
                outputs,
                sliced_labels,
                contraction_strategy="quality_multistart",
            )
            if preflight_path is not None:
                preflight_path.parent.mkdir(parents=True, exist_ok=True)
                preflight_path.write_text(
                    json.dumps(
                        {
                            "schema_version": 1,
                            "cache_key": preflight_key,
                            "sliced_labels": sliced_labels,
                            "expected_slice_tasks": expected_slice_tasks,
                            "slice_cost": slice_cost,
                        },
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n",
                    encoding="utf-8",
                )
        plan_payload[0] = (sliced_labels, expected_slice_tasks, slice_cost)
    dist.broadcast_object_list(plan_payload, src=0, device=torch.device(device))
    if plan_payload[0] is None:
        raise RuntimeError("rank-zero benchmark slicing plan was not broadcast")
    sliced_labels, expected_slice_tasks, slice_cost = plan_payload[0]
    projection_and_slice_selection_seconds = time.perf_counter() - phase_start

    def execute():
        return fq.distributed_tensor_network_amplitudes(
            circuit,
            targets,
            world_size=world_size,
            distributed_executor="torch",
            backend="nccl",
            device=device,
            dtype=dtype,
            sliced_labels=sliced_labels,
            max_slices=max(4096, expected_slice_tasks),
            max_working_set_bytes=(
                None
                if args.max_working_set_gib is None
                else int(args.max_working_set_gib * 1024**3)
            ),
            working_set_safety_factor=args.working_set_safety_factor,
            memory_calibration=memory_calibration,
            plan_cache_path=args.plan_cache_path,
        )

    for _ in range(args.warmup):
        execute()
    dist.barrier()
    torch.cuda.reset_peak_memory_stats()
    times = []
    result = None
    for _ in range(args.iterations):
        dist.barrier()
        start = time.perf_counter()
        result = execute()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    dist.barrier()
    assert result is not None
    local_record = {
        "rank": rank,
        "local_rank": local_rank,
        "times_seconds": times,
        "median_seconds": float(torch.tensor(times).median().item()),
        "peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_memory_bytes": int(torch.cuda.max_memory_reserved()),
        "device_name": torch.cuda.get_device_name(local_rank),
        "phase_timings_seconds": {
            "circuit_build": circuit_build_seconds,
            "tensor_network_build": tn_build_seconds,
            "projection_and_slice_selection": projection_and_slice_selection_seconds,
        },
    }
    records = [None for _ in range(world_size)]
    dist.all_gather_object(records, local_record)
    if rank == 0:
        values = result.values.detach().cpu()
        payload = {
            "schema_version": 1,
            "benchmark": "tn_sparse_capacity",
            "status": "measured_success",
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "git_sha": _git_sha(),
            "host": socket.gethostname(),
            "platform": platform.platform(),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "workload": {
                "topology": (
                    f"grid_{args.grid_rows}x{args.grid_cols}"
                    if grid_enabled
                    else "nearest_neighbour_brickwork"
                ),
                "qubits": n_qubits,
                "depth": args.depth,
                "target": "few_amplitudes",
                "target_count": args.targets,
                "dtype": args.dtype,
            },
            "distributed": {
                "world_size": world_size,
                "backend": "nccl",
                "sliced_labels": sliced_labels,
                "expected_slice_tasks": expected_slice_tasks,
                "estimated_per_slice_cost": slice_cost["estimated_cost"],
                "estimated_per_slice_peak": slice_cost["peak_size"],
                "max_working_set_gib": args.max_working_set_gib,
                "working_set_safety_factor": args.working_set_safety_factor,
                "plan_cache_path": (
                    None if args.plan_cache_path is None else str(args.plan_cache_path)
                ),
                "memory_calibration_identity": (
                    None if memory_calibration is None else memory_calibration.identity
                ),
            },
            "execution_summary": result.summary(),
            "phase_timings_seconds_max_rank": {
                key: max(record["phase_timings_seconds"][key] for record in records)
                for key in (
                    "circuit_build",
                    "tensor_network_build",
                    "projection_and_slice_selection",
                )
            },
            "rank_records": records,
            "latency_seconds_max_rank_median": max(
                record["median_seconds"] for record in records
            ),
            "peak_memory_bytes_max": max(
                record["peak_memory_bytes"] for record in records
            ),
            "peak_reserved_memory_bytes_max": max(
                record["peak_reserved_memory_bytes"] for record in records
            ),
            "values_real": values.real.tolist(),
            "values_imag": values.imag.tolist(),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(payload, sort_keys=True))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
