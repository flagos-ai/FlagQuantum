"""Local compiler benchmark for topology path-cache effectiveness."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from flagquantum.compiler.routing import (
    CouplingMap,
    estimate_routing_cost,
    route_to_topology,
)
from flagquantum.core.ir import CircuitIR, Instruction


def _build_workload(n_wires: int, gate_count: int) -> CircuitIR:
    pair_count = max(1, min(32, n_wires // 4))
    instructions = tuple(
        Instruction(
            "cx",
            (
                index % pair_count,
                n_wires - 1 - (index % pair_count),
            ),
        )
        for index in range(gate_count)
    )
    return CircuitIR(n_wires, instructions)


def run_benchmark(
    *,
    n_wires: int = 64,
    gate_count: int = 2000,
    path_cache_capacity: int = 4096,
    strategy: str = "restore_after_each_gate",
) -> dict[str, Any]:
    """Measure cold and warm routing of one deterministic large gate stream."""

    if n_wires < 4:
        raise ValueError("n_wires must be at least four")
    if gate_count <= 0:
        raise ValueError("gate_count must be positive")
    ir = _build_workload(n_wires, gate_count)
    estimate_coupling = CouplingMap.line(
        n_wires,
        path_cache_capacity=path_cache_capacity,
    )
    started = time.perf_counter()
    estimate = estimate_routing_cost(
        ir,
        estimate_coupling,
        strategy=strategy,
    )
    planning_seconds = time.perf_counter() - started

    coupling = CouplingMap.line(
        n_wires,
        path_cache_capacity=path_cache_capacity,
    )

    started = time.perf_counter()
    cold = route_to_topology(ir, coupling, strategy=strategy)
    cold_seconds = time.perf_counter() - started

    started = time.perf_counter()
    warm = route_to_topology(ir, coupling, strategy=strategy)
    warm_seconds = time.perf_counter() - started

    cold_routing = cold.metadata["routing"]
    warm_routing = warm.metadata["routing"]
    return {
        "schema": "flagquantum_compiler_routing_cache_benchmark_v1",
        "artifact_classification": "local_compiler_microbenchmark",
        "distribution_semantics": "single_device_fast_path",
        "scalability_claim_allowed": False,
        "n_wires": n_wires,
        "gate_count": gate_count,
        "path_cache_capacity": path_cache_capacity,
        "strategy": strategy,
        "planning_seconds": planning_seconds,
        "routing_estimate": estimate.summary(),
        "estimate_matches_materialized": (
            estimate.estimated_instruction_count == len(cold)
            and estimate.planned_inserted_swap_count
            == cold_routing["planned_inserted_swap_count"]
        ),
        "cold_seconds": cold_seconds,
        "warm_seconds": warm_seconds,
        "cold_path_cache_delta": cold_routing["path_cache"]["delta"],
        "warm_path_cache_delta": warm_routing["path_cache"]["delta"],
        "cold_instruction_count": len(cold),
        "warm_instruction_count": len(warm),
        "planned_inserted_swap_count": cold_routing["planned_inserted_swap_count"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-wires", type=int, default=64)
    parser.add_argument("--gate-count", type=int, default=2000)
    parser.add_argument("--path-cache-capacity", type=int, default=4096)
    parser.add_argument(
        "--strategy",
        choices=("restore_after_each_gate", "persistent_layout"),
        default="restore_after_each_gate",
    )
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    payload = run_benchmark(
        n_wires=args.n_wires,
        gate_count=args.gate_count,
        path_cache_capacity=args.path_cache_capacity,
        strategy=args.strategy,
    )
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
