"""Machine gate for the approved Phase 2 Batch A static compiler budget."""

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
from pathlib import Path
from typing import Any

import torch

from flagquantum._compiler.exporters.circuit_ir import seal_circuit_ir_round_trip
from flagquantum._compiler.passes.manager import PassManager
from flagquantum._compiler.passes.static_canonicalization import (
    phase2_batch_a_passes,
)
from flagquantum._compiler.testing.differential import (
    lower_module_for_differential,
)
from flagquantum.core.ir import CircuitIR, Instruction

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BUDGET = (
    ROOT / "tests/fixtures/internal_ir/phase2_performance_budget_candidate.json"
)


def build_ir(gate_count: int) -> CircuitIR:
    instructions = []
    for index in range(int(gate_count)):
        family = index % 8
        if family == 0:
            instruction = Instruction("i", (index % 4,))
        elif family in {1, 2}:
            instruction = Instruction("x", (index % 4,))
        elif family == 3:
            instruction = Instruction("rx", (index % 4,), {"theta": 0.013})
        elif family == 4:
            instruction = Instruction("rx", (index % 4,), {"theta": -0.013})
        elif family == 5:
            instruction = Instruction("cx", (index % 4, (index + 1) % 4))
        elif family == 6:
            instruction = Instruction("ry", ((index + 2) % 4,), {"theta": 0.017})
        else:
            instruction = Instruction("cz", ((index + 1) % 4, (index + 3) % 4))
        instructions.append(instruction)
    return CircuitIR(4, tuple(instructions), dtype="complex64")


def percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(float(value) for value in values)
    rank = (len(ordered) - 1) * float(fraction)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def compile_batch_a(source: CircuitIR) -> tuple[str, str, str]:
    sealed = seal_circuit_ir_round_trip(source)
    assert sealed.ok and sealed.artifact is not None
    manager = PassManager(phase2_batch_a_passes())
    pipeline = manager.run(sealed.artifact.imported.module)
    assert pipeline.ok
    lowered = lower_module_for_differential(sealed.artifact, pipeline.module)
    assert lowered.ok and lowered.circuit_ir is not None
    return (
        pipeline.module.program_identity,
        pipeline.pipeline_digest,
        lowered.circuit_ir.content_hash,
    )


def measure_case(
    budget: dict[str, Any], *, iterations: int, warmup: int
) -> dict[str, Any]:
    source = build_ir(int(budget["gate_count"]))
    for _ in range(warmup):
        compile_batch_a(source)
    timings = []
    identities = []
    gc.collect()
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(iterations):
            started = time.perf_counter_ns()
            identities.append(compile_batch_a(source))
            timings.append((time.perf_counter_ns() - started) / 1_000_000.0)
    finally:
        if gc_was_enabled:
            gc.enable()
    gc.collect()
    tracemalloc.start()
    try:
        compile_batch_a(source)
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    p95_ms = percentile(timings, 0.95)
    latency_limit = float(budget["pipeline_p95_ms_max"])
    memory_limit = int(budget["peak_host_memory_bytes_max"])
    return {
        "gate_count": int(budget["gate_count"]),
        "timings_ms": {
            "min": round(min(timings), 6),
            "p50": round(statistics.median(timings), 6),
            "p95": round(p95_ms, 6),
            "max": round(max(timings), 6),
        },
        "peak_host_memory_bytes": int(peak_bytes),
        "budget": {
            "pipeline_p95_ms_max": latency_limit,
            "peak_host_memory_bytes_max": memory_limit,
        },
        "deterministic_identity": len(set(identities)) == 1,
        "latency_passed": p95_ms <= latency_limit,
        "memory_passed": peak_bytes <= memory_limit,
    }


def evaluate(
    budget_path: Path = DEFAULT_BUDGET,
    *,
    iterations: int = 15,
    warmup: int = 3,
) -> dict[str, Any]:
    budget = json.loads(budget_path.read_text(encoding="utf-8"))
    cases = [
        measure_case(item, iterations=iterations, warmup=warmup)
        for item in budget["budgets"]
    ]
    normalized = [case["timings_ms"]["p95"] / case["gate_count"] for case in cases]
    growth_passed = all(
        normalized[index] <= normalized[index - 1] * 1.5
        for index in range(2, len(normalized))
    )
    passed = (
        all(
            case["latency_passed"]
            and case["memory_passed"]
            and case["deterministic_identity"]
            for case in cases
        )
        and growth_passed
    )
    return {
        "schema_version": "1.0",
        "status": "passed" if passed else "failed",
        "claim_scope": "Phase 2 Batch A internal local CPU gate; not a public SLA",
        "budget_path": str(budget_path.relative_to(ROOT)),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "platform": platform.platform(),
            "torch_num_threads": torch.get_num_threads(),
        },
        "method": {
            "iterations": int(iterations),
            "warmup": int(warmup),
            "clock": "time.perf_counter_ns",
            "peak_memory": "tracemalloc",
        },
        "growth_passed": growth_passed,
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget", type=Path, default=DEFAULT_BUDGET)
    parser.add_argument("--iterations", type=int, default=15)
    parser.add_argument("--warmup", type=int, default=3)
    args = parser.parse_args()
    if args.iterations < 3 or args.warmup < 1:
        raise ValueError("iterations must be >= 3 and warmup must be >= 1")
    torch.set_num_threads(1)
    payload = evaluate(args.budget, iterations=args.iterations, warmup=args.warmup)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if payload["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
