#!/usr/bin/env python3
"""Matched single-GPU FlagQuantum versus NVIDIA cuStateVec forward benchmark.

Both paths execute the same deterministic complex64 gate sequence from |0>.
This is local comparative evidence, never distributed scalability evidence.
"""

from __future__ import annotations

import argparse
import cmath
import json
import math
import platform
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import flagquantum as fq  # noqa: E402, I001
from benchmarks.sc27_metadata import (  # noqa: E402
    driver_version,
    gpu_identity,
    source_identity,
    topology_snapshot,
)


SCHEMA = "flagquantum.custatevec_statevector_compare.v1"
CUDA_C_32F = 4


@dataclass(frozen=True)
class Gate:
    name: str
    wires: tuple[int, ...]
    angle: float | None = None


def workload(n_wires: int, layers: int, seed: int = 41) -> tuple[Gate, ...]:
    gates: list[Gate] = []
    for layer in range(layers):
        for wire in range(n_wires):
            angle = 0.11 + 0.013 * ((seed + layer * n_wires + wire) % 37)
            gates.append(Gate("ry", (wire,), angle))
        edges = (
            [(wire, wire + 1) for wire in range(n_wires - 1)]
            if layer % 2 == 0
            else [(wire + 1, wire) for wire in range(n_wires - 2, -1, -1)]
        )
        gates.extend(Gate("cx", edge) for edge in edges)
    return tuple(gates)


def build_flagquantum(n_wires: int, gates: tuple[Gate, ...]) -> fq.Circuit:
    circuit = fq.Circuit(n_wires, device="cuda", dtype=torch.complex64)
    for gate in gates:
        if gate.name == "ry":
            circuit.ry(gate.wires[0], gate.angle)
        elif gate.name == "rz":
            circuit.rz(gate.wires[0], gate.angle)
        else:
            circuit.cx(*gate.wires)
    return circuit


def _matrix(cp: Any, gate: Gate) -> Any:
    if gate.name == "cx":
        return cp.asarray([[0, 1], [1, 0]], dtype=cp.complex64)
    assert gate.angle is not None
    if gate.name == "ry":
        cosine = math.cos(gate.angle / 2)
        sine = math.sin(gate.angle / 2)
        return cp.asarray([[cosine, -sine], [sine, cosine]], dtype=cp.complex64)
    phase = gate.angle / 2
    return cp.asarray(
        [[cmath.exp(-1j * phase), 0], [0, cmath.exp(1j * phase)]],
        dtype=cp.complex64,
    )


class CuStateVecExecutor:
    def __init__(self, n_wires: int, gates: tuple[Gate, ...]) -> None:
        import cupy as cp
        from cuquantum.bindings import custatevec as cusv

        self.cp = cp
        self.cusv = cusv
        self.n_wires = n_wires
        self.gates = gates
        self.handle = cusv.create()
        self.matrices = tuple(_matrix(cp, gate) for gate in gates)
        workspace_sizes = [
            int(
                cusv.apply_matrix_get_workspace_size(
                    self.handle,
                    CUDA_C_32F,
                    n_wires,
                    matrix.data.ptr,
                    CUDA_C_32F,
                    cusv.MatrixLayout.ROW,
                    0,
                    1,
                    1 if gate.name == "cx" else 0,
                    cusv.ComputeType.COMPUTE_32F,
                )
            )
            for gate, matrix in zip(gates, self.matrices, strict=True)
        ]
        self.workspace_size = max(workspace_sizes, default=0)
        self.workspace = cp.cuda.alloc(self.workspace_size) if self.workspace_size else None

    def close(self) -> None:
        if self.handle:
            self.cusv.destroy(self.handle)
            self.handle = 0

    def run(self) -> Any:
        cp, cusv = self.cp, self.cusv
        state = cp.zeros(2**self.n_wires, dtype=cp.complex64)
        state[0] = 1
        workspace_pointer = 0 if self.workspace is None else self.workspace.ptr
        for gate, matrix in zip(self.gates, self.matrices, strict=True):
            if gate.name == "cx":
                control, target = gate.wires
                controls = (self.n_wires - 1 - control,)
            else:
                target = gate.wires[0]
                controls = ()
            cusv.apply_matrix(
                self.handle,
                state.data.ptr,
                CUDA_C_32F,
                self.n_wires,
                matrix.data.ptr,
                CUDA_C_32F,
                cusv.MatrixLayout.ROW,
                0,
                (self.n_wires - 1 - target,),
                1,
                controls,
                (),
                len(controls),
                cusv.ComputeType.COMPUTE_32F,
                workspace_pointer,
                self.workspace_size,
            )
        return state


def _measure_paired(
    left: Callable[[], Any],
    right: Callable[[], Any],
    *,
    warmup: int,
    iterations: int,
    synchronize_left: Callable[[], None],
    synchronize_right: Callable[[], None],
    order_seed: int,
) -> tuple[tuple[float, ...], tuple[float, ...], Any, Any]:
    outputs: list[Any] = [None, None]
    functions = (left, right)
    synchronizers = (synchronize_left, synchronize_right)
    for index in range(warmup):
        order = (0, 1) if (index + order_seed) % 2 == 0 else (1, 0)
        for slot in order:
            outputs[slot] = functions[slot]()
            synchronizers[slot]()
    samples: list[list[float]] = [[], []]
    for index in range(iterations):
        order = (0, 1) if (index + warmup + order_seed) % 2 == 0 else (1, 0)
        for slot in order:
            synchronizers[slot]()
            started = time.perf_counter()
            outputs[slot] = functions[slot]()
            synchronizers[slot]()
            samples[slot].append(time.perf_counter() - started)
    return tuple(samples[0]), tuple(samples[1]), outputs[0], outputs[1]


def _summary(samples: tuple[float, ...]) -> dict[str, Any]:
    mean = statistics.fmean(samples)
    return {
        "samples_seconds": samples,
        "median_seconds": statistics.median(samples),
        "mean_seconds": mean,
        "coefficient_of_variation": statistics.pstdev(samples) / mean,
    }


def run_benchmark(
    *,
    n_wires: int,
    layers: int,
    seed: int,
    warmup: int,
    iterations: int,
    independent_run_index: int | None,
    container_digest: str | None,
    raw_log_sha256: str | None,
) -> dict[str, Any]:
    if n_wires < 2 or layers < 1 or warmup < 0 or iterations < 2:
        raise ValueError("n_wires >= 2, layers >= 1, warmup >= 0, iterations >= 2")
    import cupy as cp
    import cuquantum

    gates = workload(n_wires, layers, seed)
    circuit = build_flagquantum(n_wires, gates)
    executor = CuStateVecExecutor(n_wires, gates)
    try:
        torch.cuda.reset_peak_memory_stats()
        fq_samples, cusv_samples, fq_state, cusv_state = _measure_paired(
            lambda: circuit.state(refresh=True),
            executor.run,
            warmup=warmup,
            iterations=iterations,
            synchronize_left=torch.cuda.synchronize,
            synchronize_right=cp.cuda.Stream.null.synchronize,
            order_seed=seed,
        )
        fq_peak = int(torch.cuda.max_memory_allocated())
        cusv_pool_bytes = int(cp.get_default_memory_pool().total_bytes())
        reference = cp.from_dlpack(fq_state.detach())
        max_error = float(cp.max(cp.abs(reference.reshape(-1) - cusv_state)).get())
    finally:
        executor.close()
    fq_timing, cusv_timing = _summary(fq_samples), _summary(cusv_samples)
    tolerance = 3e-5
    workload_identity = {
        "name": "full_width_linear_hea_forward_state",
        "n_wires": n_wires,
        "layers": layers,
        "parameter_count": n_wires * layers,
        "seed": seed,
        "dtype": "complex64",
        "world_size": 1,
    }
    return {
        "schema": SCHEMA,
        "benchmark": "flagquantum_native_vs_custatevec_forward",
        "artifact_class": "measured_local_comparison",
        "claim_evidence_type": "local_comparative_performance",
        "comparison_class": "forward_subsystem_only",
        "distribution_semantics": "single_device_fast_path",
        "world_size": 1,
        "scalability_claim_allowed": False,
        "release_gate_allowed": False,
        "rank_placement": [
            {
                "rank": 0,
                "local_rank": 0,
                "hostname": platform.node(),
                "device": "cuda:0",
                **gpu_identity(0),
                "topology": topology_snapshot(),
            }
        ],
        "source_identity": source_identity(
            repo_root=REPO_ROOT,
            workload=workload_identity,
            container_digest=container_digest,
            raw_log_sha256=raw_log_sha256,
        ),
        "external_baseline_isolation": {
            "environment_prefix": sys.prefix,
            "flagquantum_runtime_dependency": False,
            "purpose": "benchmark_only",
            "provider": "NVIDIA cuStateVec",
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device_name": torch.cuda.get_device_name(0),
            "cuquantum": cuquantum.__version__,
            "cupy": cp.__version__,
            "driver_version": driver_version(),
        },
        "workload": {
            "name": "full_width_linear_hea_forward_state",
            "n_wires": n_wires,
            "layers": layers,
            "parameter_count": n_wires * layers,
            "gate_count": len(gates),
            "gate_set": ["RY", "CX"],
            "dtype": "complex64",
            "initial_state": "zero",
            "seed": seed,
        },
        "protocol": {
            "warmup": warmup,
            "repetitions": iterations,
            "independent_run_index": independent_run_index,
            "synchronization": "per_runtime_device_synchronized_wall_clock",
            "execution_order": "counterbalanced_by_sample",
            "setup_excluded": True,
        },
        "correctness": {
            "passed": max_error <= tolerance,
            "max_abs_error": max_error,
            "absolute_tolerance": tolerance,
        },
        "flagquantum": {**fq_timing, "peak_memory_allocated_bytes": fq_peak},
        "custatevec": {
            **cusv_timing,
            "cupy_pool_bytes": cusv_pool_bytes,
            "workspace_bytes": executor.workspace_size,
        },
        "speedup_flagquantum_over_custatevec": (
            cusv_timing["median_seconds"] / fq_timing["median_seconds"]
        ),
        "speedup_custatevec_over_flagquantum": (
            fq_timing["median_seconds"] / cusv_timing["median_seconds"]
        ),
        "stability": {
            "maximum_coefficient_of_variation": 0.10,
            "flagquantum_passed": fq_timing["coefficient_of_variation"] <= 0.10,
            "custatevec_passed": cusv_timing["coefficient_of_variation"] <= 0.10,
            "comparison_claim_allowed": (
                fq_timing["coefficient_of_variation"] <= 0.10
                and cusv_timing["coefficient_of_variation"] <= 0.10
            ),
        },
        "timing_protocol": {
            "scope": "state_initialization_plus_all_gate_applications",
            "synchronization": "device_synchronized_wall_clock",
            "execution_order": "counterbalanced_by_sample",
            "setup_excluded": [
                "circuit_construction",
                "gate_matrix_construction",
                "custatevec_handle_creation",
                "workspace_allocation",
            ],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-wires", type=int, default=31)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--independent-run-index", type=int, choices=(1, 2, 3))
    parser.add_argument("--container-digest")
    parser.add_argument("--raw-log-sha256")
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    payload = run_benchmark(
        n_wires=args.n_wires,
        layers=args.layers,
        seed=args.seed,
        warmup=args.warmup,
        iterations=args.iterations,
        independent_run_index=args.independent_run_index,
        container_digest=args.container_digest,
        raw_log_sha256=args.raw_log_sha256,
    )
    encoded = json.dumps(payload, indent=2, sort_keys=True)
    print(encoded)
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(encoded + "\n", encoding="utf-8")
    if not payload["correctness"]["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
