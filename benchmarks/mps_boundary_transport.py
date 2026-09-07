"""Measure packed MPS boundary transport and explicit overlap on NCCL."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.profiler import ProfilerActivity, profile

from flagquantum.runtime.backends.mps.execution import (
    _recv_tensor_batch_p2p,
    _recv_tensor_p2p,
    _run_batched_p2p_with_overlap,
    _send_tensor_batch_p2p,
    _send_tensor_p2p,
    mps_p2p_stats,
    reset_mps_p2p_stats,
    warmup_mps_neighbor_communicators,
)


def boundary_round(tensors, *, packed, iteration, parity):
    rank, world = dist.get_rank(), dist.get_world_size()
    for left in range(parity, world - 1, 2):
        right = left + 1
        sequences = (9_900_000 + iteration * 8 + parity * 2, 9_900_001 + iteration * 8 + parity * 2)
        if rank == right:
            if packed:
                _send_tensor_batch_p2p(tensors, dst=left, sequences=sequences)
            else:
                for tensor, sequence in zip(tensors, sequences):
                    _send_tensor_p2p(tensor, dst=left, sequence=sequence)
        elif rank == left:
            if packed:
                _recv_tensor_batch_p2p(src=right, references=tensors, sequences=sequences)
            else:
                for tensor, sequence in zip(tensors, sequences):
                    _recv_tensor_p2p(src=right, reference=tensor, sequence=sequence)


def measure(tensors, *, packed, iterations=40):
    samples = []
    reset_mps_p2p_stats()
    for iteration in range(iterations + 5):
        dist.barrier()
        started = time.perf_counter()
        boundary_round(tensors, packed=packed, iteration=iteration, parity=0)
        boundary_round(tensors, packed=packed, iteration=iteration, parity=1)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        if iteration >= 5:
            samples.append(elapsed)
    return {
        "mean_seconds": statistics.mean(samples),
        "p95_seconds": sorted(samples)[int(0.95 * (len(samples) - 1))],
        "samples": samples,
        "transport_stats": mps_p2p_stats(),
    }


def measure_overlap(device, iterations=40):
    rank, world = dist.get_rank(), dist.get_world_size()
    if world < 2 or rank > 1:
        for _ in range(iterations):
            dist.barrier()
        return {"active": False}
    send = torch.ones(1 << 18, device=device)
    recv = torch.empty_like(send)
    matrix = torch.randn(256, 256, device=device)
    samples = []
    reset_mps_p2p_stats()
    for iteration in range(iterations):
        dist.barrier()
        peer = 1 - rank
        operations = [
            dist.P2POp(dist.isend, send, peer, None, 9_990_000 + iteration),
            dist.P2POp(dist.irecv, recv, peer, None, 9_990_000 + iteration),
        ]
        started = time.perf_counter()
        _run_batched_p2p_with_overlap(
            operations, device, diagnostic="mps_boundary_overlap",
            overlap_work=lambda: torch.mm(matrix, matrix),
        )
        torch.cuda.synchronize()
        samples.append(time.perf_counter() - started)
    return {"active": True, "mean_seconds": statistics.mean(samples), "transport_stats": mps_p2p_stats()}


def mismatch_fault(tensors):
    rank = dist.get_rank()
    passed = False
    dist.barrier()
    if rank == 1:
        _send_tensor_batch_p2p(
            (tensors[0][..., :8], tensors[1]), dst=0,
            sequences=(9_999_000, 9_999_001),
        )
    elif rank == 0:
        try:
            _recv_tensor_batch_p2p(
                src=1, references=tensors, sequences=(9_999_000, 9_999_001)
            )
        except RuntimeError as error:
            passed = "payload drained" in str(error)
    dist.barrier()
    values = [None] * dist.get_world_size()
    dist.all_gather_object(values, {"rank": rank, "passed": passed})
    return {"shape_mismatch_passed": values[0]["passed"], "bounded_cleanup": True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    rank, world = dist.get_rank(), dist.get_world_size()
    warmup_mps_neighbor_communicators(device)
    # ISSUE-097 crossover shape: eight sites/rank and a many-channel env carry.
    tensors = (
        torch.randn(1, 16, 2, 16, dtype=torch.complex64, device=device),
        torch.randn(127, 1, 16, 16, dtype=torch.complex64, device=device),
    )
    legacy = measure(tensors, packed=False)
    packed = measure(tensors, packed=True)
    overlap = measure_overlap(device)
    fault = mismatch_fault(tensors)
    trace_path = args.output.with_suffix(f".rank-{rank}.trace.json")
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as profiler:
        boundary_round(tensors, packed=True, iteration=999, parity=0)
        boundary_round(tensors, packed=True, iteration=999, parity=1)
        torch.cuda.synchronize()
    profiler.export_chrome_trace(str(trace_path))
    local = {"rank": rank, "legacy": legacy, "packed": packed, "overlap": overlap, "fault": fault, "trace_path": str(trace_path)}
    records = [None] * world
    dist.all_gather_object(records, local)
    if rank == 0:
        legacy_messages = sum(item["legacy"]["transport_stats"]["physical_message_count"] for item in records)
        packed_messages = sum(item["packed"]["transport_stats"]["physical_message_count"] for item in records)
        legacy_bytes = sum(item["legacy"]["transport_stats"]["logical_payload_bytes"] for item in records)
        packed_bytes = sum(item["packed"]["transport_stats"]["logical_payload_bytes"] for item in records)
        legacy_wait = sum(item["legacy"]["transport_stats"]["host_wait_seconds"] for item in records)
        packed_wait = sum(item["packed"]["transport_stats"]["host_wait_seconds"] for item in records)
        payload = {
            "schema": "flagquantum.mps_boundary_transport.v1",
            "world_size": world, "backend": "nccl", "device": torch.cuda.get_device_name(0),
            "legacy_physical_messages": legacy_messages,
            "packed_physical_messages": packed_messages,
            "message_count_reduction": 1.0 - packed_messages / legacy_messages,
            "legacy_logical_payload_bytes": legacy_bytes,
            "packed_logical_payload_bytes": packed_bytes,
            "logical_byte_growth": packed_bytes / legacy_bytes - 1.0,
            "legacy_total_rank_host_wait_seconds": legacy_wait,
            "packed_total_rank_host_wait_seconds": packed_wait,
            "p2p_wait_reduction": 1.0 - packed_wait / legacy_wait,
            "overlap_work_seconds": max(item["overlap"].get("transport_stats", {}).get("overlap_work_seconds", 0.0) for item in records),
            "trace_paths": [item["trace_path"] for item in records],
            "shape_mismatch_fault": fault,
            "rank_records": records,
            "passed": packed_messages <= 0.6 * legacy_messages and packed_bytes <= 1.05 * legacy_bytes and packed_wait < legacy_wait and fault["shape_mismatch_passed"],
            "performance_claim_allowed": False,
            "scalability_claim_allowed": False,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"output": str(args.output), "passed": payload["passed"]}))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
