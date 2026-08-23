"""Matched A800 benchmark for owner-sharded MPS parameter synchronization."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import torch
import torch.distributed as dist

from flagquantum.runtime.backends.mps.training_engine import (
    _broadcast_parameters,
    _parameter_broadcast_buckets,
)


def _measure(operation, *, warmup: int, repetitions: int, device: torch.device):
    samples = []
    for iteration in range(warmup + repetitions):
        dist.barrier()
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        operation()
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        maximum = torch.tensor(elapsed, dtype=torch.float64, device=device)
        dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
        if iteration >= warmup:
            samples.append(float(maximum.cpu()))
    return samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parameters", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--bucket-bytes", type=int, default=25 * 1024 * 1024)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    dist.init_process_group("nccl")
    rank, world_size = dist.get_rank(), dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", world_size))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    owners = tuple(index % world_size for index in range(args.parameters))

    def fresh_parameters():
        return tuple(
            torch.tensor(
                float(index + 1) if owners[index] == rank else -1.0,
                dtype=torch.float64,
                device=device,
            )
            for index in range(args.parameters)
        )

    legacy_parameters = fresh_parameters()

    def legacy():
        for parameter, owner in zip(legacy_parameters, owners):
            dist.broadcast(parameter, src=owner)

    legacy_samples = _measure(
        legacy, warmup=args.warmup, repetitions=args.repetitions, device=device
    )
    bucketed_parameters = fresh_parameters()
    buckets = _parameter_broadcast_buckets(
        bucketed_parameters,
        owners,
        max_bucket_bytes=args.bucket_bytes,
        world_size=world_size,
        rank=rank,
    )
    bucketed_samples = _measure(
        lambda: _broadcast_parameters(bucketed_parameters, buckets),
        warmup=args.warmup,
        repetitions=args.repetitions,
        device=device,
    )
    expected = torch.arange(
        1, args.parameters + 1, dtype=torch.float64, device=device
    )
    actual = torch.stack(bucketed_parameters)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    if rank == 0:
        legacy_median = statistics.median(legacy_samples)
        bucketed_median = statistics.median(bucketed_samples)
        payload = {
            "schema": "flagquantum.mps_parameter_broadcast_ab.v1",
            "device_name": torch.cuda.get_device_name(device),
            "backend": "nccl",
            "world_size": world_size,
            "local_world_size": local_world_size,
            "node_count": world_size // local_world_size,
            "parameter_count": args.parameters,
            "parameter_owner_policy": "round_robin_unique_owner",
            "dtype": "float64",
            "legacy_collective_count": args.parameters,
            "bucketed_collective_count": len(buckets),
            "legacy_median_seconds": legacy_median,
            "bucketed_median_seconds": bucketed_median,
            "speedup": legacy_median / bucketed_median,
            "exact_parameter_parity": True,
            "distribution_semantics": "sharded_across_ranks",
            "communication_protocol": "bounded_dtype_padded_all_gather_single",
            "scalability_claim_allowed": False,
            "scalability_blockers": [
                "specialized_parameter_synchronization_microbenchmark",
                "end_to_end_scaling_evidence_not_attached",
            ],
            "release_gate_allowed": False,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, sort_keys=True) + "\n")
        print(json.dumps(payload, sort_keys=True), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
