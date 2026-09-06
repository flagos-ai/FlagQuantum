"""Measure private Phase 2 Batch E identity and in-memory pipeline cache."""

from __future__ import annotations

import argparse
import gc
import json
import math
import platform
import time
import tracemalloc
from collections.abc import Callable, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from flagquantum._compiler.exporters.circuit_ir import seal_circuit_ir_round_trip
from flagquantum._compiler.ir.modules import QuantumModule
from flagquantum._compiler.passes.manager import PassManager
from flagquantum._compiler.passes.static_canonicalization import (
    StaticCanonicalizationPass,
)
from flagquantum._compiler.pipeline_cache import (
    BoundedPipelineCache,
    CacheDisposition,
    CachedPipelineRunner,
    CompilationIdentityInputs,
)
from flagquantum.core.ir import CircuitIR, Instruction

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BUDGET = (
    ROOT
    / "tests"
    / "fixtures"
    / "internal_ir"
    / "phase2_batch_e_performance_budget.json"
)


def build_module(gate_count: int) -> QuantumModule:
    templates = (
        Instruction("rx", (0,), {"theta": 0.13}),
        Instruction("ry", (3,), {"theta": -0.17}),
        Instruction("rz", (5,), {"theta": 0.19}),
        Instruction("cx", (0, 7)),
        Instruction("x", (2,)),
        Instruction("x", (2,)),
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


def _inputs(*, cacheable: bool = True) -> CompilationIdentityInputs:
    return CompilationIdentityInputs(
        source_identity="a" * 64,
        target_profile="universal_rx_ry_rz_cx_v1",
        topology_identity="b" * 64 if cacheable else None,
        calibration_identity="c" * 64,
        compile_options={"optimization_level": 2},
    )


def _time(
    operation: Callable[[], str], iterations: int
) -> tuple[list[float], list[str]]:
    timings = []
    identities = []
    for _ in range(iterations):
        started = time.perf_counter_ns()
        identities.append(operation())
        timings.append((time.perf_counter_ns() - started) / 1_000_000.0)
    return timings, identities


def measure_case(gate_count: int, *, iterations: int, warmup: int) -> dict[str, Any]:
    module = build_module(gate_count)
    manager = PassManager((StaticCanonicalizationPass(),))

    def cold() -> str:
        result = CachedPipelineRunner(manager, BoundedPipelineCache()).run(
            module, _inputs()
        )
        assert result.ok and result.identity is not None
        assert result.cache_disposition is CacheDisposition.MISS
        return result.identity.digest

    hit_cache = BoundedPipelineCache()
    hit_runner = CachedPipelineRunner(manager, hit_cache)
    seeded = hit_runner.run(module, _inputs())
    assert seeded.ok and seeded.identity is not None

    def hit() -> str:
        result = hit_runner.run(module, _inputs())
        assert result.ok and result.identity is not None
        assert result.cache_disposition is CacheDisposition.HIT
        return result.identity.digest

    bypass_runner = CachedPipelineRunner(manager, BoundedPipelineCache())

    def bypass() -> str:
        result = bypass_runner.run(module, _inputs(cacheable=False))
        assert result.ok and result.pipeline is not None
        assert result.cache_disposition is CacheDisposition.BYPASS
        return result.pipeline.module.program_identity

    for _ in range(warmup):
        cold()
        hit()
        bypass()
    gc.collect()
    cold_timings, cold_identities = _time(cold, iterations)
    hit_timings, hit_identities = _time(hit, iterations)
    bypass_timings, bypass_identities = _time(bypass, iterations)
    gc.collect()
    tracemalloc.start()
    try:
        cold()
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    cold_p95 = percentile(cold_timings, 0.95)
    hit_p95 = percentile(hit_timings, 0.95)
    return {
        "gate_count": gate_count,
        "cold_miss_p95_ms": round(cold_p95, 6),
        "cache_hit_p95_ms": round(hit_p95, 6),
        "identity_bypass_p95_ms": round(percentile(bypass_timings, 0.95), 6),
        "observed_hit_speedup": round(cold_p95 / hit_p95, 6),
        "cold_peak_host_memory_bytes": int(peak_bytes),
        "deterministic_identity": (
            len(set(cold_identities + hit_identities)) == 1
            and len(set(bypass_identities)) == 1
        ),
        "cache_snapshot": asdict(hit_cache.snapshot()),
    }


def evaluate(budget_path: Path, *, iterations: int, warmup: int) -> dict[str, Any]:
    budget_path = budget_path.resolve()
    payload = json.loads(budget_path.read_text(encoding="utf-8"))
    budgets = {int(item["gate_count"]): item for item in payload["budgets"]}
    cases = []
    for gate_count in (10, 100, 1000, 10000):
        case = measure_case(gate_count, iterations=iterations, warmup=warmup)
        budget = budgets[gate_count]
        case["budget"] = budget
        case["cold_miss_latency_passed"] = (
            case["cold_miss_p95_ms"] <= budget["cold_miss_p95_ms_max"]
        )
        case["cache_hit_latency_passed"] = (
            case["cache_hit_p95_ms"] <= budget["cache_hit_p95_ms_max"]
        )
        case["bypass_latency_passed"] = (
            case["identity_bypass_p95_ms"] <= budget["identity_bypass_p95_ms_max"]
        )
        case["hit_speedup_passed"] = (
            case["observed_hit_speedup"] >= budget["cache_hit_speedup_min"]
        )
        case["memory_passed"] = (
            case["cold_peak_host_memory_bytes"]
            <= budget["cold_peak_host_memory_bytes_max"]
        )
        case["passed"] = bool(
            case["cold_miss_latency_passed"]
            and case["cache_hit_latency_passed"]
            and case["bypass_latency_passed"]
            and case["hit_speedup_passed"]
            and case["memory_passed"]
            and case["deterministic_identity"]
        )
        cases.append(case)
    passed = all(case["passed"] for case in cases)
    return {
        "schema_version": "1.0",
        "status": "passed" if passed else "failed",
        "claim_scope": "Phase 2 Batch E private CPU budget; not a public SLA",
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
            "pipeline": "one private static-canonicalization pass",
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
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
