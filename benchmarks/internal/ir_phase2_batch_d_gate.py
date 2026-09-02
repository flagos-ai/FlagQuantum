"""Measure private Phase 2 Batch D deterministic text emission."""

from __future__ import annotations

import argparse
import gc
import json
import math
import platform
import statistics
import time
import tracemalloc
from collections.abc import Callable, Sequence
from typing import Any

import torch

from flagquantum._compiler.exporters.circuit_ir import seal_circuit_ir_round_trip
from flagquantum._compiler.exporters.text import (
    TextEmissionResult,
    emit_openqasm2,
    emit_openqasm3,
    emit_qcis_v1,
)
from flagquantum.core.ir import CircuitIR, Instruction

Emitter = Callable[[object], TextEmissionResult]
EMITTERS: tuple[tuple[str, Emitter], ...] = (
    ("openqasm2", emit_openqasm2),
    ("openqasm3", emit_openqasm3),
    ("qcis_v1", emit_qcis_v1),
)


def build_module(gate_count: int) -> object:
    templates = (
        Instruction("rx", (0,), {"theta": 0.13}),
        Instruction("ry", (3,), {"theta": -0.17}),
        Instruction("rz", (5,), {"theta": 0.19}),
        Instruction("cx", (0, 7)),
    )
    source = CircuitIR(
        8,
        tuple(templates[index % len(templates)] for index in range(gate_count)),
    )
    sealed = seal_circuit_ir_round_trip(source)
    assert sealed.ok and sealed.artifact is not None
    return sealed.artifact.imported.module


def percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(float(value) for value in values)
    rank = (len(ordered) - 1) * float(fraction)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def measure_emitter(
    module: object,
    emitter: Emitter,
    *,
    iterations: int,
    warmup: int,
) -> dict[str, Any]:
    for _ in range(warmup):
        assert emitter(module).ok
    timings = []
    hashes = []
    gc.collect()
    for _ in range(iterations):
        started = time.perf_counter_ns()
        result = emitter(module)
        timings.append((time.perf_counter_ns() - started) / 1_000_000.0)
        assert result.ok and result.content_hash is not None
        hashes.append(result.content_hash)
    gc.collect()
    tracemalloc.start()
    try:
        observed = emitter(module)
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert observed.ok and observed.text is not None
    return {
        "timings_ms": {
            "p50": round(statistics.median(timings), 6),
            "p95": round(percentile(timings, 0.95), 6),
            "max": round(max(timings), 6),
        },
        "peak_host_memory_bytes": int(peak_bytes),
        "output_bytes": len(observed.text.encode("utf-8")),
        "deterministic_content_hash": len(set(hashes)) == 1,
    }


def evaluate(*, iterations: int, warmup: int) -> dict[str, Any]:
    cases = []
    for gate_count in (10, 100, 1000, 10000):
        module = build_module(gate_count)
        measurements = {
            name: measure_emitter(
                module,
                emitter,
                iterations=iterations,
                warmup=warmup,
            )
            for name, emitter in EMITTERS
        }
        cases.append({"gate_count": gate_count, "emitters": measurements})
    return {
        "schema_version": "1.0",
        "status": "measured",
        "claim_scope": "Phase 2 Batch D private CPU baseline; not a public SLA",
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "platform": platform.platform(),
            "torch_num_threads": torch.get_num_threads(),
        },
        "method": {
            "iterations": iterations,
            "warmup": warmup,
            "clock": "time.perf_counter_ns",
            "peak_memory": "tracemalloc",
            "scope": "emission of a prebuilt verified native target module",
        },
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    args = parser.parse_args()
    if args.iterations < 3 or args.warmup < 1:
        raise ValueError("iterations must be >= 3 and warmup must be >= 1")
    torch.set_num_threads(1)
    print(
        json.dumps(
            evaluate(iterations=args.iterations, warmup=args.warmup),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
