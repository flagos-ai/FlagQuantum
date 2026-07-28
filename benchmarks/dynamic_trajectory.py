"""Development benchmark for dynamic trajectory execution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

import flagquantum as fq


def _workload(mid_circuit_measurements: int) -> fq.experimental.DynamicCircuit:
    if mid_circuit_measurements <= 0:
        raise ValueError("mid_circuit_measurements must be positive")
    circuit = fq.experimental.DynamicCircuit(2)
    circuit.h(0)
    for classical_bit in range(mid_circuit_measurements):
        circuit.measure(0, classical_bit=classical_bit)
        circuit.conditional("x", 1, classical_bit=classical_bit, equals=1)
    return circuit


def _executor(name: str) -> Callable[..., fq.experimental.DynamicExecutionResult]:
    if name == "flagquantum":
        return fq.experimental.run_dynamic
    if name == "qiskit_aer":
        return fq.experimental.run_qiskit_aer_dynamic
    raise ValueError(f"unknown backend: {name}")


def run_benchmark(
    *,
    shots: tuple[int, ...] = (100, 1000),
    mid_circuit_measurements: tuple[int, ...] = (1, 2, 4),
    backends: tuple[str, ...] = ("flagquantum",),
    seed: int = 7,
    flagquantum_strategy: str = "auto",
) -> dict[str, Any]:
    """Measure shot throughput without making production performance claims."""

    if not shots or any(value <= 0 for value in shots):
        raise ValueError("shots must contain positive integers")
    rows = []
    for backend in backends:
        execute = _executor(backend)
        for measurement_count in mid_circuit_measurements:
            circuit = _workload(measurement_count)
            for shot_count in shots:
                started = perf_counter()
                options = {"shots": shot_count, "seed": seed}
                if backend == "flagquantum":
                    options["strategy"] = flagquantum_strategy
                result = execute(circuit, **options)
                elapsed = perf_counter() - started
                unique_branches = len(
                    {
                        tuple(row)
                        for row in result.classical_bits.reshape(
                            -1, result.classical_bits.shape[-1]
                        ).tolist()
                    }
                )
                rows.append(
                    {
                        "backend": backend,
                        "shots": shot_count,
                        "mid_circuit_measurements": measurement_count,
                        "elapsed_seconds": elapsed,
                        "shots_per_second": shot_count / elapsed,
                        "observed_branch_count": unique_branches,
                        "execution_semantics": result.execution_semantics,
                        "runtime_statistics": dict(result.statistics),
                    }
                )
    return {
        "schema": "flagquantum_dynamic_trajectory_benchmark_v1",
        "artifact_classification": "development_microbenchmark",
        "scalability_claim_allowed": False,
        "seed": seed,
        "flagquantum_strategy": flagquantum_strategy,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shots", type=int, nargs="+", default=(100, 1000))
    parser.add_argument(
        "--mid-circuit-measurements",
        type=int,
        nargs="+",
        default=(1, 2, 4),
    )
    parser.add_argument(
        "--backend",
        choices=("flagquantum", "qiskit_aer"),
        action="append",
        dest="backends",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--flagquantum-strategy",
        choices=("auto", "trajectory", "batched"),
        default="auto",
    )
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    payload = run_benchmark(
        shots=tuple(args.shots),
        mid_circuit_measurements=tuple(args.mid_circuit_measurements),
        backends=tuple(args.backends or ("flagquantum",)),
        seed=args.seed,
        flagquantum_strategy=args.flagquantum_strategy,
    )
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
