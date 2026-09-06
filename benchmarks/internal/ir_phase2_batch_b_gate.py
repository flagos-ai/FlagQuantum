"""Measure the private Phase 2 Batch B target-decomposition pipeline."""

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
    StaticCanonicalizationPass,
)
from flagquantum._compiler.passes.target_decomposition import (
    DecomposeToTargetGateSetPass,
)
from flagquantum._compiler.testing.differential import lower_module_for_differential
from flagquantum.core.ir import CircuitIR, Instruction

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BUDGET = (
    ROOT
    / "tests"
    / "fixtures"
    / "internal_ir"
    / "phase2_batch_b_performance_budget.json"
)


def build_ir(gate_count: int) -> CircuitIR:
    templates = (
        Instruction("h", (0,)),
        Instruction("u3", (1,), {"theta": 0.13, "phi": -0.21, "lbd": 0.34}),
        Instruction("cy", (0, 1)),
        Instruction("cz", (1, 2)),
        Instruction("swap", (0, 2)),
        Instruction("crx", (0, 1), {"theta": 0.17}),
        Instruction("cry", (1, 2), {"theta": -0.19}),
        Instruction("crz", (2, 0), {"theta": 0.23}),
        Instruction("cphase", (0, 2), {"theta": -0.29}),
        Instruction("rxx", (0, 1), {"theta": 0.31}),
        Instruction("ryy", (1, 2), {"theta": -0.37}),
        Instruction("rzz", (2, 0), {"theta": 0.41}),
        Instruction("ccx", (0, 1, 2)),
        Instruction("cswap", (0, 1, 2)),
        Instruction("sx", (1,)),
        Instruction("tdg", (2,)),
    )
    return CircuitIR(
        3,
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


def compile_batch_b(source: CircuitIR) -> tuple[str, str, str, int]:
    sealed = seal_circuit_ir_round_trip(source)
    assert sealed.ok and sealed.artifact is not None
    pipeline = PassManager(
        (StaticCanonicalizationPass(), DecomposeToTargetGateSetPass())
    ).run(sealed.artifact.imported.module)
    assert pipeline.ok
    lowered = lower_module_for_differential(sealed.artifact, pipeline.module)
    assert lowered.ok and lowered.circuit_ir is not None
    return (
        pipeline.module.program_identity,
        pipeline.pipeline_digest,
        lowered.circuit_ir.content_hash,
        len(lowered.circuit_ir.instructions),
    )


def measure_case(gate_count: int, *, iterations: int, warmup: int) -> dict[str, Any]:
    source = build_ir(gate_count)
    for _ in range(warmup):
        compile_batch_b(source)
    timings = []
    identities = []
    gc.collect()
    for _ in range(iterations):
        started = time.perf_counter_ns()
        identities.append(compile_batch_b(source))
        timings.append((time.perf_counter_ns() - started) / 1_000_000.0)
    gc.collect()
    tracemalloc.start()
    try:
        observed = compile_batch_b(source)
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return {
        "gate_count": int(gate_count),
        "emitted_gate_count": observed[3],
        "expansion_ratio": round(observed[3] / gate_count, 6),
        "timings_ms": {
            "p50": round(statistics.median(timings), 6),
            "p95": round(percentile(timings, 0.95), 6),
            "max": round(max(timings), 6),
        },
        "peak_host_memory_bytes": int(peak_bytes),
        "deterministic_identity": len(set(identities)) == 1,
    }


def evaluate(budget_path: Path, *, iterations: int, warmup: int) -> dict[str, Any]:
    budget_path = budget_path.resolve()
    budget_payload = json.loads(budget_path.read_text(encoding="utf-8"))
    budgets = {int(item["gate_count"]): item for item in budget_payload["budgets"]}
    cases = []
    for gate_count in (10, 100, 1000, 10000):
        case = measure_case(gate_count, iterations=iterations, warmup=warmup)
        budget = budgets[gate_count]
        case["budget"] = budget
        case["latency_passed"] = (
            case["timings_ms"]["p95"] <= budget["pipeline_p95_ms_max"]
        )
        case["memory_passed"] = (
            case["peak_host_memory_bytes"] <= budget["peak_host_memory_bytes_max"]
        )
        case["passed"] = bool(
            case["latency_passed"]
            and case["memory_passed"]
            and case["deterministic_identity"]
        )
        cases.append(case)
    passed = all(case["passed"] for case in cases)
    return {
        "schema_version": "1.0",
        "status": "passed" if passed else "failed",
        "claim_scope": "Phase 2 Batch B private CPU budget; not a public SLA",
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
