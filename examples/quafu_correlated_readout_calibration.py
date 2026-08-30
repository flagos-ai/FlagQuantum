"""Measure the correlated two-qubit readout matrix on Quafu hardware."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import flagquantum as fq


def preparation(bits: str) -> fq.Circuit:
    circuit = fq.Circuit(len(bits))
    for wire, bit in enumerate(bits):
        if bit == "1":
            circuit.x(wire)
    return circuit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="Baihua")
    parser.add_argument("--shots", type=int, default=10240)
    parser.add_argument("--target-qubits", type=int, nargs=2, default=(123, 124))
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/development/quafu_vqe_10240/correlated_readout.json"),
    )
    args = parser.parse_args()
    if not os.getenv("QPU_API_TOKEN"):
        raise RuntimeError("QPU_API_TOKEN is required")

    provider = fq.QuafuProvider(result_timeout=args.timeout, poll_interval=5)
    backend = next(
        item
        for item in provider.discover_backends(n_wires=2)
        if item.name == args.backend
    )
    prepared_results = {}
    states = ("00", "01", "10", "11")
    for prepared in states:
        package = fq.create_deployment_package(
            preparation(prepared),
            backend=backend,
            name=f"flagquantum_readout_q123_q124_{prepared}",
            shots=args.shots,
            metadata={
                "provider_compile": True,
                "provider_options": {
                    "compiler": "quarkcircuit",
                    "correct": False,
                    "open_dd": None,
                    "target_qubits": list(args.target_qubits),
                },
            },
        )
        result = provider.run(package)
        raw_counts = {state: int(result.counts.get(state, 0)) for state in states}
        # Quafu serializes classical bits as c1c0, whereas FlagQuantum's
        # probability basis and this file's state labels use q0q1.
        counts = {state: raw_counts[state[::-1]] for state in states}
        prepared_results[prepared] = {
            "task_id": result.handle.task_id,
            "counts": counts,
            "raw_quafu_counts_c1c0": raw_counts,
        }
        print(f"prepared={prepared} task_id={result.handle.task_id} counts={counts}")

    matrix = []
    for prepared in states:
        counts = prepared_results[prepared]["counts"]
        total = sum(counts.values())
        matrix.append([counts[observed] / total for observed in states])
    payload = {
        "schema": "flagquantum_quafu_correlated_readout_v1",
        "backend": args.backend,
        "physical_qubits": list(args.target_qubits),
        "shots_per_state": args.shots,
        "state_order": list(states),
        "prepared_results": prepared_results,
        "correlated_readout_confusion_matrix": matrix,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"saved={args.output}")


if __name__ == "__main__":
    main()
