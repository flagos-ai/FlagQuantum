"""Run a reproducible variance pilot and export its backend-selection evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import flagquantum as fq


def _circuit(n_wires: int, depth: int, device: str) -> fq.Circuit:
    circuit = fq.Circuit(n_wires, device=device)
    for layer in range(depth):
        for wire in range(n_wires):
            circuit.ry(wire, theta=0.031 * (layer + 1) * (wire + 1))
        for wire in range(layer % 2, n_wires - 1, 2):
            circuit.cx(wire, wire + 1)
    return circuit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-wires", type=int, default=8)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--pilot-trajectories", type=int, default=32)
    parser.add_argument("--trajectory-cap", type=int, default=4096)
    parser.add_argument("--trajectory-batch-size", type=int, default=32)
    parser.add_argument("--min-trajectories", type=int, default=32)
    parser.add_argument("--target-standard-error", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--pilot-confidence-level", type=float, default=0.95)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    circuit = _circuit(args.n_wires, args.depth, args.device)
    noise = (
        fq.NoiseModel()
        .add("ry", fq.amplitude_damping_channel(0.002))
        .add("cx", fq.depolarizing_channel(0.005))
    )
    pilot = fq.run_noisy_statevector(
        circuit,
        noise,
        trajectories=args.pilot_trajectories,
        trajectory_batch_size=args.trajectory_batch_size,
        seed=args.seed,
    )
    pilot_variance = float(torch.max(pilot.variance).item())
    selection = fq.plan_noise_execution_selection(
        circuit,
        noise,
        trajectories=args.trajectory_cap,
        trajectory_batch_size=args.trajectory_batch_size,
        min_trajectories=args.min_trajectories,
        target_standard_error=args.target_standard_error,
        pilot_variance=pilot_variance,
        pilot_trajectories=pilot.statistics.count,
        pilot_confidence_level=args.pilot_confidence_level,
        pilot_observable_count=pilot.variance.numel(),
        calibration=args.calibration,
    )
    payload = {
        "schema": "flagquantum.noise_pilot_selection_evidence.v1",
        "circuit_digest": circuit.to_ir().content_hash,
        "noise_model_identity": noise.identity,
        "seed": args.seed,
        "pilot_trajectory_ids": pilot.trajectory_ids,
        "pilot_expectation_z": pilot.expectation_z.cpu().tolist(),
        "pilot_variance": pilot.variance.cpu().tolist(),
        "maximum_pilot_variance": pilot_variance,
        **selection.summary(),
    }
    rendered = json.dumps(payload, indent=2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
