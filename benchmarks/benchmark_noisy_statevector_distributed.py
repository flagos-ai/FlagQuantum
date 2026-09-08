"""NCCL trajectory-parallel benchmark for batched noisy statevectors."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import statistics
import time
from pathlib import Path

import torch
import torch.distributed as dist

import flagquantum as fq
import flagquantum.noise as fqn


def _circuit(n_wires: int, depth: int, *, device: torch.device) -> fq.Circuit:
    circuit = fq.Circuit(n_wires, device=device)
    for layer in range(depth):
        for wire in range(n_wires):
            circuit.ry(wire, theta=0.07 * (layer + 1) * (wire + 1))
        for wire in range(layer % 2, n_wires - 1, 2):
            circuit.cx(wire, wire + 1)
    return circuit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-wires", type=int, default=12)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--trajectories", type=int, default=512)
    parser.add_argument("--trajectory-batch-size", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--min-trajectories", type=int, default=1)
    parser.add_argument("--target-standard-error", type=float)
    parser.add_argument("--checkpoint-path")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-batches-per-run", type=int)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--inject-failure-rank", type=int)
    parser.add_argument("--inject-failure-trajectory", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", device_id=device)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    circuit = _circuit(args.n_wires, args.depth, device=device)
    noise = (
        fqn.NoiseModel()
        .add("ry", fqn.amplitude_damping_channel(0.002))
        .add("cx", fqn.depolarizing_channel(0.005))
    )
    if args.inject_failure_rank == rank:
        noisy_backend = importlib.import_module(
            "flagquantum.runtime.executors.statevector.noisy"
        )
        original_execute = noisy_backend._execute_trajectory_batch

        def injected_execute(initial, ids, **kwargs):
            if args.inject_failure_trajectory in ids:
                raise RuntimeError("injected benchmark trajectory failure")
            return original_execute(initial, ids, **kwargs)

        noisy_backend._execute_trajectory_batch = injected_execute

    elapsed_values = []
    local_result = None
    local_peaks = []
    for _ in range(args.repeats):
        dist.barrier()
        torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        local_result = fq.run(
            circuit,
            noise_model=noise,
            mode="auto",
            trajectories=args.trajectories,
            trajectory_batch_size=args.trajectory_batch_size,
            seed=args.seed,
            min_trajectories=args.min_trajectories,
            target_standard_error=args.target_standard_error,
            checkpoint_path=args.checkpoint_path,
            resume=args.resume,
            max_batches_per_run=args.max_batches_per_run,
            continue_on_error=args.continue_on_error,
        )
        torch.cuda.synchronize(device)
        elapsed = torch.tensor(time.perf_counter() - started, device=device)
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
        elapsed_values.append(float(elapsed.item()))
        peak = torch.tensor(
            [
                torch.cuda.max_memory_allocated(device),
                torch.cuda.max_memory_reserved(device),
            ],
            dtype=torch.int64,
            device=device,
        )
        dist.all_reduce(peak, op=dist.ReduceOp.MAX)
        local_peaks.append(tuple(int(value) for value in peak.cpu().tolist()))
    assert local_result is not None

    median_seconds = statistics.median(elapsed_values)
    completed_trajectories = local_result.statistics.count

    if rank == 0:
        payload = {
            "schema": "flagquantum.noisy_statevector_distributed_benchmark.v1",
            "torch_version": torch.__version__,
            "device_name": torch.cuda.get_device_name(device),
            "world_size": world_size,
            "n_wires": args.n_wires,
            "depth": args.depth,
            "circuit_digest": circuit.to_ir().content_hash,
            "channel_count": sum(
                item.metadata.get("is_channel", False)
                for item in fq.lower_noise_model(circuit, noise)
            ),
            "requested_trajectories": args.trajectories,
            "completed_trajectories": completed_trajectories,
            "trajectory_batch_size": args.trajectory_batch_size,
            "seed": args.seed,
            "noise_model_identity": noise.identity,
            "median_seconds": median_seconds,
            "trajectories_per_second": completed_trajectories / median_seconds,
            "repeat_seconds": elapsed_values,
            "max_cuda_peak_allocated_bytes": max(item[0] for item in local_peaks),
            "max_cuda_peak_reserved_bytes": max(item[1] for item in local_peaks),
            "global_expectation_z": local_result.expectation_z.cpu().tolist(),
            "global_variance": local_result.variance.cpu().tolist(),
            "min_trajectories": args.min_trajectories,
            "target_standard_error": args.target_standard_error,
            "converged": local_result.converged,
            "stopped_early": local_result.stopped_early,
            "failures": [item.summary() for item in local_result.failures],
        }
        rendered = json.dumps(payload, indent=2)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print(rendered)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
