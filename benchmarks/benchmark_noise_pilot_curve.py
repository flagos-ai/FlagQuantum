"""Measure how pilot size tightens noisy-selector variance evidence."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

import flagquantum as fq
import flagquantum.noise as fqn


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
    parser.add_argument(
        "--pilot-sizes", type=int, nargs="+", default=(32, 128, 512, 2048)
    )
    parser.add_argument("--n-wires", type=int, default=8)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--trajectory-cap", type=int, default=4096)
    parser.add_argument("--trajectory-batch-size", type=int, default=64)
    parser.add_argument("--target-standard-error", type=float, default=0.02)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    circuit = _circuit(args.n_wires, args.depth, args.device)
    noise = (
        fqn.NoiseModel()
        .add("ry", fqn.amplitude_damping_channel(0.002))
        .add("cx", fqn.depolarizing_channel(0.005))
    )
    records = []
    for pilot_size in args.pilot_sizes:
        torch.cuda.synchronize() if torch.device(args.device).type == "cuda" else None
        started = time.perf_counter()
        pilot = fq.run_noisy_statevector(
            circuit,
            noise,
            trajectories=pilot_size,
            trajectory_batch_size=args.trajectory_batch_size,
            seed=args.seed,
        )
        torch.cuda.synchronize() if torch.device(args.device).type == "cuda" else None
        pilot_seconds = time.perf_counter() - started
        sample_variance = float(torch.max(pilot.variance).item())
        selection = fq.plan_noise_execution_selection(
            circuit,
            noise,
            trajectories=args.trajectory_cap,
            trajectory_batch_size=args.trajectory_batch_size,
            min_trajectories=pilot_size,
            target_standard_error=args.target_standard_error,
            pilot_variance=sample_variance,
            pilot_trajectories=pilot_size,
            pilot_confidence_level=args.confidence_level,
            pilot_observable_count=pilot.variance.numel(),
            calibration=args.calibration,
        )
        candidate_times = {
            item.mode: (item.metadata or {}).get("calibrated_estimated_seconds")
            for item in selection.candidates
        }
        records.append(
            {
                "pilot_trajectories": pilot_size,
                "pilot_seconds": pilot_seconds,
                "maximum_sample_variance": sample_variance,
                "variance_upper_confidence_bound": (
                    selection.trajectory_variance_estimate
                ),
                "estimated_trajectories_to_target": (
                    selection.estimated_trajectories_to_target
                ),
                "selected_mode": selection.selected_mode,
                "candidate_estimated_seconds": candidate_times,
            }
        )
    payload = {
        "schema": "flagquantum.noise_pilot_curve.v1",
        "device_name": (
            torch.cuda.get_device_name()
            if torch.cuda.is_available()
            else str(args.device)
        ),
        "torch_version": torch.__version__,
        "circuit_digest": circuit.to_ir().content_hash,
        "noise_model_identity": noise.identity,
        "n_wires": args.n_wires,
        "depth": args.depth,
        "seed": args.seed,
        "trajectory_batch_size": args.trajectory_batch_size,
        "trajectory_cap": args.trajectory_cap,
        "target_standard_error": args.target_standard_error,
        "confidence_level": args.confidence_level,
        "observable_count": args.n_wires,
        "records": records,
    }
    rendered = json.dumps(payload, indent=2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
