"""Bounded large-payload NCCL/RDMA preflight for multi-node TN execution."""

from __future__ import annotations

import argparse
import datetime
import json
import os
import socket
import time

import torch
import torch.distributed as dist


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload-mib", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--timeout-seconds", type=int, default=180)
    return parser.parse_args()


def main() -> None:
    arguments = _arguments()
    if arguments.payload_mib <= 0:
        raise ValueError("payload-mib must be positive")
    if arguments.warmup < 0 or arguments.iterations <= 0:
        raise ValueError("warmup must be non-negative and iterations positive")
    local_rank = int(os.environ["LOCAL_RANK"])
    local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group(
        "nccl",
        device_id=device,
        timeout=datetime.timedelta(seconds=arguments.timeout_seconds),
    )
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    element_count = arguments.payload_mib * 1024 * 1024 // 4
    payload = torch.ones(element_count, dtype=torch.float32, device=device)

    def execute() -> float:
        payload.fill_(1.0)
        dist.barrier()
        started = time.perf_counter()
        dist.all_reduce(payload)
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        expected = float(world_size)
        if float(payload[0]) != expected or float(payload[-1]) != expected:
            raise RuntimeError("large-payload all-reduce produced an invalid checksum")
        return elapsed

    for _ in range(arguments.warmup):
        execute()
    times = [execute() for _ in range(arguments.iterations)]
    local = {
        "rank": rank,
        "local_rank": local_rank,
        "hostname": socket.gethostname(),
        "device": torch.cuda.get_device_name(device),
        "payload_bytes": payload.numel() * payload.element_size(),
        "times_seconds": times,
        "median_seconds": float(torch.tensor(times).median()),
        "minimum_seconds": min(times),
        "maximum_seconds": max(times),
        "checksum": float(payload[0]),
    }
    records: list[dict | None] = [None] * world_size
    dist.all_gather_object(records, local)
    if rank == 0:
        hosts = sorted({str(record["hostname"]) for record in records if record})
        maximum_median = max(
            float(record["median_seconds"]) for record in records if record
        )
        payload_bytes = int(local["payload_bytes"])
        print(
            json.dumps(
                {
                    "schema": "flagquantum.tn_multinode_rdma_preflight.v1",
                    "world_size": world_size,
                    "local_world_size": local_world_size,
                    "node_count": len(hosts),
                    "hosts": hosts,
                    "backend": "nccl",
                    "payload_bytes": payload_bytes,
                    "warmup": arguments.warmup,
                    "iterations": arguments.iterations,
                    "maximum_rank_median_seconds": maximum_median,
                    "algorithmic_payload_gib_per_second": (
                        payload_bytes / maximum_median / 1024**3
                    ),
                    "rank_records": records,
                    "collective_correctness_passed": True,
                    "rdma_transport_requires_nccl_log_confirmation": True,
                    "scalability_claim_allowed": False,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
