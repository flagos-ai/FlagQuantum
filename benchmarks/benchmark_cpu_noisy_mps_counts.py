"""Benchmark low-bond noisy-MPS counts width on one CPU process."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from datetime import date
from pathlib import Path

import torch

import flagquantum as fq
import flagquantum.noise as fqn
from flagquantum.core.ir import MeasurementNode
from flagquantum.runtime.execution import run_advanced


def _ghz_chain(n_qubits: int) -> fq.Circuit:
    circuit = fq.Circuit(n_qubits).h(0)
    for wire in range(n_qubits - 1):
        circuit.cx(wire, wire + 1)
    return circuit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--widths",
        type=int,
        nargs="+",
        default=(16, 20, 24, 50, 100, 200, 500, 1000),
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--trajectories", type=int, default=4)
    parser.add_argument("--shots", type=int, default=1000)
    parser.add_argument("--max-bond", type=int, default=8)
    parser.add_argument("--cutoff", type=float, default=1e-10)
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if any(width <= 0 for width in args.widths):
        raise ValueError("widths must be positive")
    if args.repeats <= 0 or args.trajectories <= 0 or args.shots <= 0:
        raise ValueError("repeats, trajectories, and shots must be positive")

    model = fqn.NoiseModel().add("cx", fqn.depolarizing_channel(0.01))
    records = []
    for n_qubits in args.widths:
        print(f"benchmarking {n_qubits} qubits...", flush=True)
        circuit = _ghz_chain(n_qubits)
        seconds = []
        observed_bonds = []
        maximum_truncation_errors = []
        planned_state_bytes = []
        for _ in range(args.repeats):
            started = time.perf_counter()
            result = run_advanced(
                circuit,
                mode="noisy_mps",
                noise_model=model,
                device="cpu",
                trajectories=args.trajectories,
                max_bond=args.max_bond,
                cutoff=args.cutoff,
                seed=args.seed,
                measurements=(
                    MeasurementNode(
                        "counts",
                        tuple(range(n_qubits)),
                        shots=args.shots,
                        metadata={"seed": 23},
                    ),
                ),
            )
            seconds.append(time.perf_counter() - started)
            if sum(result.counts[0].values()) != args.shots:
                raise RuntimeError("counts total does not match requested shots")
            statistics_payload = result.measurement("counts").statistics
            observed_bonds.append(statistics_payload["observed_max_bond"])
            maximum_truncation_errors.append(
                statistics_payload["max_trajectory_truncation_error"]
            )
            noisy_plan = result.plan.noisy_execution_plan
            if noisy_plan is None:
                raise RuntimeError("benchmark did not produce a noisy execution plan")
            planned_state_bytes.append(noisy_plan.memory.estimated_bytes)
        records.append(
            {
                "n_qubits": n_qubits,
                "seconds": seconds,
                "median_seconds": statistics.median(seconds),
                "maximum_observed_bond": max(observed_bonds),
                "maximum_truncation_error": max(maximum_truncation_errors),
                "maximum_planned_state_bytes": max(planned_state_bytes),
                "counts_total": args.shots,
            }
        )
        print(
            f"  median={records[-1]['median_seconds']:.3f}s, "
            f"max_bond={records[-1]['maximum_observed_bond']}",
            flush=True,
        )

    payload = {
        "benchmark": "cpu_noisy_mps_counts_width",
        "benchmark_evidence_class": "local_non_release",
        "claim_evidence_type": "development_smoke",
        "distribution_semantics": "single_device_fast_path",
        "non_release_evidence": True,
        "scalability_claim_allowed": False,
        "release_gate_allowed": False,
        "scalability_blockers": [
            "local CPU benchmark is not release-certified scalability evidence"
        ],
        "schema": "flagquantum.benchmark.cpu_noisy_mps_counts",
        "version": "1.1",
        "recorded_at": date.today().isoformat(),
        "environment": {
            "os": platform.platform(),
            "architecture": platform.machine(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": "cpu",
        },
        "workload": {
            "circuit": "GHZ nearest-neighbor chain",
            "noise": "cx depolarizing(0.01)",
            "trajectories": args.trajectories,
            "shots": args.shots,
            "max_bond": args.max_bond,
            "cutoff": args.cutoff,
            "repeats": args.repeats,
        },
        "results": records,
        "scope": (
            "Low-bond CPU width evidence for this workload; not a general "
            "simulator throughput or arbitrary-circuit capacity claim"
        ),
    }
    rendered = json.dumps(payload, indent=2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
