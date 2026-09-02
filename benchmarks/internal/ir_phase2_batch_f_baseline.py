"""Measure the private Phase 2 Batch F end-to-end offline pipeline."""

from __future__ import annotations

import argparse
import gc
import json
import math
import platform
import time
import tracemalloc
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import torch

from flagquantum._compiler.offline_deployment import (
    OfflineStaticTarget,
    compile_offline_static,
)
from flagquantum._compiler.passes.placement_routing import DirectedCouplingGraph
from flagquantum._compiler.pipeline_cache import BoundedPipelineCache, CacheDisposition
from flagquantum.core.ir import CircuitIR, Instruction

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BUDGET = (
    ROOT
    / "tests"
    / "fixtures"
    / "internal_ir"
    / "phase2_batch_f_performance_budget_candidate.json"
)


def build_source(gate_count: int) -> CircuitIR:
    templates = (
        Instruction("rx", (0,), {"theta": 0.13}),
        Instruction("h", (2,)),
        Instruction("ry", (3,), {"theta": -0.17}),
        Instruction("cx", (3, 4)),
        Instruction("rz", (5,), {"theta": 0.19}),
        Instruction("cx", (6, 7)),
    )
    return CircuitIR(
        8,
        tuple(templates[index % len(templates)] for index in range(gate_count)),
        dtype="complex128",
    )


def build_target() -> OfflineStaticTarget:
    edges = tuple(
        edge for left in range(7) for edge in ((left, left + 1), (left + 1, left))
    )
    return OfflineStaticTarget(DirectedCouplingGraph(8, edges), "a" * 64)


def percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(float(value) for value in values)
    rank = (len(ordered) - 1) * float(fraction)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _time(operation: Callable[[], tuple[str, int]], iterations: int):
    timings = []
    identities = []
    text_bytes = []
    for _ in range(iterations):
        started = time.perf_counter_ns()
        identity, size = operation()
        timings.append((time.perf_counter_ns() - started) / 1_000_000.0)
        identities.append(identity)
        text_bytes.append(size)
    return timings, identities, text_bytes


def measure_case(gate_count: int, *, iterations: int, warmup: int) -> dict[str, Any]:
    source = build_source(gate_count)
    target = build_target()

    def cold() -> tuple[str, int]:
        result = compile_offline_static(
            source, target, cache=BoundedPipelineCache(max_entries=1)
        )
        assert result.ok and result.execution is not None
        assert result.execution.cache_disposition is CacheDisposition.MISS
        return result.execution.identity.digest, sum(
            len(emission.text.encode("utf-8")) for _, emission in result.emissions
        )

    cache = BoundedPipelineCache(max_entries=1)
    seeded = compile_offline_static(source, target, cache=cache)
    assert seeded.ok

    def cached() -> tuple[str, int]:
        result = compile_offline_static(source, target, cache=cache)
        assert result.ok and result.execution is not None
        assert result.execution.cache_disposition is CacheDisposition.HIT
        return result.execution.identity.digest, sum(
            len(emission.text.encode("utf-8")) for _, emission in result.emissions
        )

    for _ in range(warmup):
        cold()
        cached()
    gc.collect()
    cold_timings, cold_identities, cold_bytes = _time(cold, iterations)
    cached_timings, cached_identities, cached_bytes = _time(cached, iterations)
    gc.collect()
    tracemalloc.start()
    try:
        cold()
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return {
        "gate_count": gate_count,
        "cold_end_to_end_p95_ms": round(percentile(cold_timings, 0.95), 6),
        "cached_end_to_end_p95_ms": round(percentile(cached_timings, 0.95), 6),
        "cold_peak_host_memory_bytes": int(peak_bytes),
        "emitted_text_bytes": cold_bytes[0],
        "deterministic_identity": len(set(cold_identities + cached_identities)) == 1,
        "deterministic_text_size": len(set(cold_bytes + cached_bytes)) == 1,
    }


def evaluate(budget_path: Path, *, iterations: int, warmup: int) -> dict[str, Any]:
    budget_path = budget_path.resolve()
    payload = json.loads(budget_path.read_text(encoding="utf-8"))
    budgets = {int(item["gate_count"]): item for item in payload["budgets"]}
    cases = []
    for count in (10, 100, 1000, 10000):
        case = measure_case(count, iterations=iterations, warmup=warmup)
        budget = budgets[count]
        case["budget"] = budget
        case["cold_latency_passed"] = (
            case["cold_end_to_end_p95_ms"] <= budget["cold_end_to_end_p95_ms_max"]
        )
        case["cached_latency_passed"] = (
            case["cached_end_to_end_p95_ms"] <= budget["cached_end_to_end_p95_ms_max"]
        )
        case["memory_passed"] = (
            case["cold_peak_host_memory_bytes"]
            <= budget["cold_peak_host_memory_bytes_max"]
        )
        case["passed"] = bool(
            case["cold_latency_passed"]
            and case["cached_latency_passed"]
            and case["memory_passed"]
            and case["deterministic_identity"]
            and case["deterministic_text_size"]
        )
        cases.append(case)
    passed = all(case["passed"] for case in cases)
    return {
        "schema_version": "1.0",
        "status": "passed" if passed else "failed",
        "claim_scope": "Phase 2 Batch F private CPU budget; not a public SLA",
        "budget_path": str(budget_path.relative_to(ROOT)),
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
            "pipeline": "import, canonicalize, decompose, route, cache, and three emitters",
        },
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--budget", type=Path, default=DEFAULT_BUDGET)
    args = parser.parse_args()
    if args.iterations < 3 or args.warmup < 1:
        raise ValueError("iterations must be >= 3 and warmup must be >= 1")
    torch.set_num_threads(1)
    result = evaluate(args.budget, iterations=args.iterations, warmup=args.warmup)
    print(json.dumps(result, indent=2))
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
