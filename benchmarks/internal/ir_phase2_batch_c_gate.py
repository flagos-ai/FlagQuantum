"""Measure the private Phase 2 Batch C placement/routing pipeline."""

from __future__ import annotations

import argparse
import gc
import json
import math
import platform
import statistics
import time
import tracemalloc
from collections.abc import Sequence
from typing import Any

import torch

from flagquantum._compiler.exporters.circuit_ir import seal_circuit_ir_round_trip
from flagquantum._compiler.passes.manager import PassManager
from flagquantum._compiler.passes.placement_routing import (
    DirectedCouplingGraph,
    PlacementRoutingPass,
)
from flagquantum._compiler.testing.differential import lower_module_for_differential
from flagquantum.core.ir import CircuitIR, Instruction


def line_graph(n_qubits: int) -> DirectedCouplingGraph:
    return DirectedCouplingGraph(
        n_qubits,
        tuple(
            edge
            for left in range(n_qubits - 1)
            for edge in ((left, left + 1), (left + 1, left))
        ),
    )


def build_ir(gate_count: int) -> CircuitIR:
    templates = (
        Instruction("rx", (0,), {"theta": 0.13}),
        Instruction("cx", (0, 7)),
        Instruction("cx", (6, 1)),
        Instruction("cx", (2, 5)),
        Instruction("cx", (3, 4)),
        Instruction("ry", (3,), {"theta": -0.17}),
        Instruction("cx", (0, 4)),
        Instruction("cx", (1, 5)),
        Instruction("cx", (2, 6)),
        Instruction("cx", (3, 7)),
        Instruction("rz", (5,), {"theta": 0.19}),
        Instruction("cx", (0, 2)),
        Instruction("cx", (1, 3)),
        Instruction("cx", (4, 6)),
        Instruction("cx", (5, 7)),
    )
    return CircuitIR(
        8,
        tuple(templates[index % len(templates)] for index in range(gate_count)),
        dtype="complex64",
    )


def percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(float(value) for value in values)
    rank = (len(ordered) - 1) * float(fraction)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def compile_batch_c(source: CircuitIR) -> tuple[str, str, str, int, int]:
    sealed = seal_circuit_ir_round_trip(source)
    assert sealed.ok and sealed.artifact is not None
    pipeline = PassManager((PlacementRoutingPass(line_graph(source.n_wires)),)).run(
        sealed.artifact.imported.module
    )
    assert pipeline.ok
    lowered = lower_module_for_differential(sealed.artifact, pipeline.module)
    assert lowered.ok and lowered.circuit_ir is not None
    routing = pipeline.pass_results[-1].statistics
    return (
        pipeline.module.program_identity,
        pipeline.pipeline_digest,
        lowered.circuit_ir.content_hash,
        len(lowered.circuit_ir.instructions),
        int(routing["physical_swap_count"]),
    )


def measure_case(gate_count: int, *, iterations: int, warmup: int) -> dict[str, Any]:
    source = build_ir(gate_count)
    for _ in range(warmup):
        compile_batch_c(source)
    timings = []
    identities = []
    gc.collect()
    for _ in range(iterations):
        started = time.perf_counter_ns()
        identities.append(compile_batch_c(source))
        timings.append((time.perf_counter_ns() - started) / 1_000_000.0)
    gc.collect()
    tracemalloc.start()
    try:
        observed = compile_batch_c(source)
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return {
        "gate_count": gate_count,
        "emitted_gate_count": observed[3],
        "physical_swap_count": observed[4],
        "expansion_ratio": round(observed[3] / gate_count, 6),
        "timings_ms": {
            "p50": round(statistics.median(timings), 6),
            "p95": round(percentile(timings, 0.95), 6),
            "max": round(max(timings), 6),
        },
        "peak_host_memory_bytes": int(peak_bytes),
        "deterministic_identity": len(set(identities)) == 1,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    args = parser.parse_args()
    if args.iterations < 3 or args.warmup < 1:
        raise ValueError("iterations must be >= 3 and warmup must be >= 1")
    torch.set_num_threads(1)
    cases = [
        measure_case(gates, iterations=args.iterations, warmup=args.warmup)
        for gates in (10, 100, 1000, 10000)
    ]
    print(
        json.dumps(
            {
                "schema_version": "1.0",
                "status": "measured",
                "claim_scope": "Phase 2 Batch C private CPU baseline; not a public SLA",
                "environment": {
                    "python": platform.python_version(),
                    "torch": torch.__version__,
                    "platform": platform.platform(),
                    "torch_num_threads": torch.get_num_threads(),
                },
                "method": {
                    "iterations": args.iterations,
                    "warmup": args.warmup,
                    "clock": "time.perf_counter_ns",
                    "peak_memory": "tracemalloc",
                    "topology": "8-qubit bidirectional line",
                    "workload": "native long-range CX-heavy deterministic mix",
                },
                "cases": cases,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
