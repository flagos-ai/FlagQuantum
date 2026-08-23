"""Export a reproducible noisy-backend selector decision as JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import flagquantum as fq


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-wires", type=int, default=12)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--theta-scale", type=float, default=0.031)
    parser.add_argument("--trajectories", type=int, default=512)
    parser.add_argument("--trajectory-batch-size", type=int, default=64)
    parser.add_argument("--min-trajectories", type=int, default=1)
    parser.add_argument("--target-standard-error", type=float)
    parser.add_argument("--pilot-variance", type=float)
    parser.add_argument("--pilot-trajectories", type=int)
    parser.add_argument("--pilot-confidence-level", type=float)
    parser.add_argument("--pilot-observable-count", type=int, default=1)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--memory-limit-bytes", type=int, default=80 * 1024**3)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--disallow-approximate", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    circuit = fq.Circuit(args.n_wires)
    for layer in range(args.depth):
        for wire in range(args.n_wires):
            circuit.ry(wire, theta=args.theta_scale * (layer + 1) * (wire + 1))
        for wire in range(layer % 2, args.n_wires - 1, 2):
            circuit.cx(wire, wire + 1)
    noise = (
        fq.NoiseModel()
        .add("ry", fq.amplitude_damping_channel(0.002))
        .add("cx", fq.depolarizing_channel(0.005))
    )
    selection = fq.plan_noise_execution_selection(
        circuit,
        noise,
        world_size=args.world_size,
        trajectories=args.trajectories,
        trajectory_batch_size=args.trajectory_batch_size,
        memory_limit_bytes=args.memory_limit_bytes,
        calibration=args.calibration,
        min_trajectories=args.min_trajectories,
        target_standard_error=args.target_standard_error,
        pilot_variance=args.pilot_variance,
        pilot_trajectories=args.pilot_trajectories,
        pilot_confidence_level=args.pilot_confidence_level,
        pilot_observable_count=args.pilot_observable_count,
        allow_approximate=not args.disallow_approximate,
    )
    payload = {
        "schema": "flagquantum.noise_execution_selection_evidence.v1",
        "circuit_digest": circuit.to_ir().content_hash,
        "noise_model_identity": noise.identity,
        **selection.summary(),
    }
    rendered = json.dumps(payload, indent=2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
