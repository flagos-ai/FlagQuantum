"""Collect A800 calibration data for the noisy backend selector."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

import flagquantum as fq
import flagquantum.backends as fqb
import flagquantum.noise as fqn


def _circuit(n_wires: int, depth: int, *, device: str) -> fq.Circuit:
    circuit = fq.Circuit(n_wires, device=device)
    for layer in range(depth):
        for wire in range(n_wires):
            circuit.ry(wire, theta=0.031 * (layer + 1) * (wire + 1))
        for wire in range(layer % 2, n_wires - 1, 2):
            circuit.cx(wire, wire + 1)
    return circuit


def _noise(kind: str) -> fqn.NoiseModel:
    model = fqn.NoiseModel().add("ry", fqn.amplitude_damping_channel(0.002))
    if kind == "full":
        model.add("cx", fqn.depolarizing_channel(0.005))
    return model


def _run_once(
    circuit: fq.Circuit,
    noise: fqn.NoiseModel,
    mode: str,
    *,
    trajectories: int,
    trajectory_batch_size: int,
    mps_trajectories: int,
) -> object:
    if mode == "noisy_statevector":
        return fqb.run_native(
            circuit,
            noise_model=noise,
            mode=mode,
            trajectories=trajectories,
            trajectory_batch_size=trajectory_batch_size,
            seed=20260806,
        )
    if mode == "noisy_mps":
        return fqb.run_native(
            circuit,
            noise_model=noise,
            mode=mode,
            trajectories=min(trajectories, mps_trajectories),
            max_bond=64,
            cutoff=1e-8,
            seed=20260806,
            retain_trajectories=False,
        )
    return fqb.run_native(circuit, noise_model=noise, mode="density_matrix")


def _measure(
    circuit: fq.Circuit,
    noise: fqn.NoiseModel,
    mode: str,
    *,
    trajectories: int,
    trajectory_batch_size: int,
    mps_trajectories: int,
    repeats: int,
) -> dict[str, object]:
    _run_once(
        circuit,
        noise,
        mode,
        trajectories=trajectories,
        trajectory_batch_size=trajectory_batch_size,
        mps_trajectories=mps_trajectories,
    )
    timings = []
    peaks = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        _run_once(
            circuit,
            noise,
            mode,
            trajectories=trajectories,
            trajectory_batch_size=trajectory_batch_size,
            mps_trajectories=mps_trajectories,
        )
        torch.cuda.synchronize()
        timings.append(time.perf_counter() - started)
        peaks.append(torch.cuda.max_memory_allocated())
    return {
        "mode": mode,
        "median_seconds": statistics.median(timings),
        "repeat_seconds": timings,
        "max_cuda_peak_allocated_bytes": max(peaks),
        "executed_trajectories": (
            None
            if mode == "density_matrix"
            else min(trajectories, mps_trajectories)
            if mode == "noisy_mps"
            else trajectories
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qubits", type=int, nargs="+", default=(8, 10, 12, 16))
    parser.add_argument("--depths", type=int, nargs="+", default=(4, 8))
    parser.add_argument("--noise-kinds", nargs="+", default=("local", "full"))
    parser.add_argument("--trajectories", type=int, default=128)
    parser.add_argument("--trajectory-batch-size", type=int, default=64)
    parser.add_argument("--mps-trajectories", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if torch.device(args.device).type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("selector calibration requires an available CUDA device")

    records = []
    for n_wires in args.qubits:
        for depth in args.depths:
            circuit = _circuit(n_wires, depth, device=args.device)
            for noise_kind in args.noise_kinds:
                noise = _noise(noise_kind)
                lowered = fq.lower_noise_model(circuit, noise)
                channel_count = sum(
                    item.metadata.get("is_channel", False) for item in lowered
                )
                modes = ["noisy_statevector"]
                if n_wires <= 8:
                    modes.extend(("density_matrix", "noisy_mps"))
                for mode in modes:
                    measured = _measure(
                        circuit,
                        noise,
                        mode,
                        trajectories=args.trajectories,
                        trajectory_batch_size=args.trajectory_batch_size,
                        mps_trajectories=args.mps_trajectories,
                        repeats=args.repeats,
                    )
                    records.append(
                        {
                            "n_wires": n_wires,
                            "depth": depth,
                            "noise_kind": noise_kind,
                            "channel_count": channel_count,
                            "circuit_digest": circuit.to_ir().content_hash,
                            "noise_model_identity": noise.identity,
                            **measured,
                        }
                    )
    payload = {
        "schema": "flagquantum.noise_selector_calibration.v1",
        "torch_version": torch.__version__,
        "device_name": torch.cuda.get_device_name(),
        "trajectory_batch_size": args.trajectory_batch_size,
        "requested_trajectories": args.trajectories,
        "world_size": 1,
        "records": records,
    }
    rendered = json.dumps(payload, indent=2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
