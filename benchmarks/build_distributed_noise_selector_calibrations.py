"""Convert distributed trajectory benchmarks into selector calibrations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import flagquantum as fq
import flagquantum.noise as fqn


def _circuit(n_wires: int, depth: int) -> fq.Circuit:
    circuit = fq.Circuit(n_wires)
    for layer in range(depth):
        for wire in range(n_wires):
            circuit.ry(wire, theta=0.07 * (layer + 1) * (wire + 1))
        for wire in range(layer % 2, n_wires - 1, 2):
            circuit.cx(wire, wire + 1)
    return circuit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for source in args.inputs:
        benchmark = json.loads(source.read_text(encoding="utf-8"))
        if benchmark.get("schema") != (
            "flagquantum.noisy_statevector_distributed_benchmark.v1"
        ):
            raise ValueError(f"unsupported benchmark schema in {source}")
        n_wires = int(benchmark["n_wires"])
        depth = int(benchmark["depth"])
        circuit = _circuit(n_wires, depth)
        noise = (
            fqn.NoiseModel()
            .add("ry", fqn.amplitude_damping_channel(0.002))
            .add("cx", fqn.depolarizing_channel(0.005))
        )
        if noise.identity != benchmark["noise_model_identity"]:
            raise ValueError(f"noise identity mismatch in {source}")
        lowered = fq.lower_noise_model(circuit, noise)
        channel_count = sum(item.metadata.get("is_channel", False) for item in lowered)
        trajectories = int(
            benchmark.get("requested_trajectories", benchmark.get("trajectories"))
        )
        world_size = int(benchmark["world_size"])
        payload = {
            "schema": fq.NOISE_SELECTOR_CALIBRATION_SCHEMA,
            "torch_version": benchmark["torch_version"],
            "device_name": benchmark["device_name"],
            "trajectory_batch_size": int(benchmark["trajectory_batch_size"]),
            "requested_trajectories": trajectories,
            "world_size": world_size,
            "records": [
                {
                    "n_wires": n_wires,
                    "depth": depth,
                    "noise_kind": "full",
                    "channel_count": channel_count,
                    "circuit_digest": circuit.to_ir().content_hash,
                    "noise_model_identity": noise.identity,
                    "mode": "noisy_statevector",
                    "median_seconds": float(benchmark["median_seconds"]),
                    "max_cuda_peak_allocated_bytes": int(
                        benchmark["max_cuda_peak_allocated_bytes"]
                    ),
                    "executed_trajectories": int(
                        benchmark.get("completed_trajectories", trajectories)
                    ),
                }
            ],
            "source_benchmark": str(source),
        }
        output = args.output_dir / (
            f"noise_selector_calibration_a800_{world_size}gpu_20260806.json"
        )
        output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(output)


if __name__ == "__main__":
    main()
