"""Benchmark true batched statevector trajectories on one reproducible workload."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

import flagquantum as fq
import flagquantum.noise as fqn


def _circuit(n_wires: int, depth: int, *, device: str) -> fq.Circuit:
    circuit = fq.Circuit(n_wires, device=device)
    for layer in range(depth):
        for wire in range(n_wires):
            circuit.ry(wire, theta=0.07 * (layer + 1) * (wire + 1))
        offset = layer % 2
        for wire in range(offset, n_wires - 1, 2):
            circuit.cx(wire, wire + 1)
    return circuit


def _measure(
    circuit: fq.Circuit,
    noise: fqn.NoiseModel,
    *,
    trajectories: int,
    trajectory_batch_size: int,
    seed: int,
    repeats: int,
) -> tuple[dict[str, object], torch.Tensor]:
    device = torch.device(circuit.device)
    timings = []
    result = None
    peak_allocated = None
    peak_reserved = None
    for _ in range(repeats):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        result = fq.run_noisy_statevector(
            circuit,
            noise,
            trajectories=trajectories,
            trajectory_batch_size=trajectory_batch_size,
            seed=seed,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            peak_allocated = torch.cuda.max_memory_allocated(device)
            peak_reserved = torch.cuda.max_memory_reserved(device)
        timings.append(time.perf_counter() - started)
    assert result is not None
    median_seconds = statistics.median(timings)
    return (
        {
            "trajectory_batch_size": trajectory_batch_size,
            "median_seconds": median_seconds,
            "trajectories_per_second": trajectories / median_seconds,
            "repeat_seconds": timings,
            "cuda_peak_allocated_bytes": peak_allocated,
            "cuda_peak_reserved_bytes": peak_reserved,
        },
        result.expectation_z.detach().cpu(),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-wires", type=int, default=12)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--trajectories", type=int, default=256)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=(1, 8, 32, 64))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    circuit = _circuit(args.n_wires, args.depth, device=args.device)
    noise = (
        fqn.NoiseModel()
        .add("ry", fqn.amplitude_damping_channel(0.002))
        .add("cx", fqn.depolarizing_channel(0.005))
    )
    rows = []
    reference = None
    maximum_expectation_difference = 0.0
    for batch_size in args.batch_sizes:
        row, expectation = _measure(
            circuit,
            noise,
            trajectories=args.trajectories,
            trajectory_batch_size=batch_size,
            seed=args.seed,
            repeats=args.repeats,
        )
        if reference is None:
            reference = expectation
        else:
            difference = float(torch.max(torch.abs(reference - expectation)).item())
            maximum_expectation_difference = max(
                maximum_expectation_difference, difference
            )
            if not torch.allclose(reference, expectation, atol=2e-6, rtol=2e-6):
                raise RuntimeError(
                    "seeded result changed beyond floating-point tolerance with "
                    f"trajectory batch size; max_abs_difference={difference:.3e}"
                )
        rows.append(row)
    baseline = float(rows[0]["trajectories_per_second"])
    for row in rows:
        row["speedup_vs_first"] = float(row["trajectories_per_second"]) / baseline
    payload = {
        "schema": "flagquantum.noisy_statevector_benchmark.v1",
        "torch_version": torch.__version__,
        "device": args.device,
        "device_name": (
            torch.cuda.get_device_name(torch.device(args.device))
            if torch.device(args.device).type == "cuda"
            else str(torch.device(args.device))
        ),
        "n_wires": args.n_wires,
        "depth": args.depth,
        "trajectories": args.trajectories,
        "seed": args.seed,
        "noise_model_identity": noise.identity,
        "maximum_expectation_difference": maximum_expectation_difference,
        "results": rows,
    }
    rendered = json.dumps(payload, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
