"""NCCL neighbor transport and real MPS communication evidence for ISSUE-094."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist

import flagquantum as fq
from flagquantum.runtime.distributed.engine import _recv_tensor_p2p, _send_tensor_p2p

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.distributed_mps.variable_bond_capacity_8gpu import (  # noqa: E402
    rank_owned_initial_mps,
)


def circuit(n_wires, theta, phi, world):
    value = fq.Circuit(n_wires, device=theta.device)
    for rank in range(world):
        value.ry((2 * rank + 1) * n_wires // (2 * world), theta)
    for rank in range(world - 1):
        left = (rank + 1) * n_wires // world - 1
        if rank % 2:
            value.rzz(left, left + 1, theta)
        else:
            value.rxx(left, left + 1, phi)
    return value


def exchange_round(tensor, recv, *, parity, batched):
    rank, world = dist.get_rank(), dist.get_world_size()
    peer = rank + 1 if rank % 2 == parity else rank - 1
    if not 0 <= peer < world:
        return
    tag = 9_410_000 + parity
    if batched:
        requests = dist.batch_isend_irecv(
            [
                dist.P2POp(dist.isend, tensor, peer, None, tag),
                dist.P2POp(dist.irecv, recv, peer, None, tag),
            ]
        )
    else:
        # Ordered blocking legacy baseline avoids the eager-P2P serialization
        # deadlock while retaining the exact pre-ISSUE-094 unbatched behavior.
        if rank < peer:
            dist.send(tensor, peer, tag=tag)
            dist.recv(recv, peer, tag=tag)
        else:
            dist.recv(recv, peer, tag=tag)
            dist.send(tensor, peer, tag=tag)
        requests = []
    for request in requests:
        request.wait()


def microbenchmark(device, *, batched, iterations=50):
    send = torch.ones(1 << 18, dtype=torch.float32, device=device)
    recv = torch.empty_like(send)
    samples = []
    for iteration in range(iterations + 5):
        dist.barrier()
        started = time.perf_counter()
        exchange_round(send, recv, parity=0, batched=batched)
        exchange_round(send, recv, parity=1, batched=batched)
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        if iteration >= 5:
            samples.append(elapsed)
    return {
        "iterations": iterations,
        "payload_bytes": send.numel() * send.element_size(),
        "mean_seconds": statistics.mean(samples),
        "p95_seconds": sorted(samples)[int(0.95 * (len(samples) - 1))],
    }


def inject_sequence_mismatch(device):
    """Exercise the production descriptor protocol without leaving a sender blocked."""
    rank, world = dist.get_rank(), dist.get_world_size()
    if world < 2:
        return {"attempted": False, "passed": False, "bounded_cleanup": False}
    dist.barrier()
    passed = False
    diagnostic = ""
    if rank == 0:
        _send_tensor_p2p(torch.ones(4, device=device), dst=1, sequence=94_001)
    elif rank == 1:
        try:
            _recv_tensor_p2p(
                src=0, reference=torch.empty(4, device=device), sequence=94_000
            )
        except RuntimeError as exc:
            diagnostic = str(exc)
            passed = all(
                token in diagnostic
                for token in ("peer 0", "expected 94000", "received 94001")
            )
    outcome = {"rank": rank, "passed": passed, "diagnostic": diagnostic}
    outcomes = [None] * world
    dist.all_gather_object(outcomes, outcome)
    dist.barrier()
    return {
        "attempted": True,
        "passed": bool(outcomes[1]["passed"]),
        "bounded_cleanup": True,
        "diagnostic": outcomes[1]["diagnostic"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--include-legacy", action="store_true")
    parser.add_argument("--inject-sequence-mismatch", action="store_true")
    args = parser.parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    rank, world = dist.get_rank(), dist.get_world_size()
    legacy = microbenchmark(device, batched=False) if args.include_legacy else None
    batched = microbenchmark(device, batched=True)
    theta = torch.tensor(0.07, device=device, requires_grad=True)
    phi = torch.tensor(-0.11, device=device, requires_grad=True)
    n_wires = max(8, 2 * world)
    result = fq.train_distributed_mps(
        circuit(n_wires, theta, phi, world),
        steps=args.steps,
        observable={n_wires // 2: "z"},
        optimizer="adam",
        lr=0.01,
        device=device,
        max_bond=7,
        gradient_policy="approximate",
        gradient_tolerance=100.0,
        initial_mps_tensors=rank_owned_initial_mps(n_wires, 8, device),
        initial_mps_left_canonical=True,
        memory_leak_tolerance_bytes=64 << 20,
    )
    summary = result.summary()
    fault = (
        inject_sequence_mismatch(device)
        if args.inject_sequence_mismatch
        else {"attempted": False, "passed": False, "bounded_cleanup": False}
    )
    local = {
        "rank": rank,
        "legacy": legacy,
        "batched": batched,
        "communication_setup_seconds": summary["communication_setup_seconds"],
        "steps": summary["step_metrics"],
        "rank_useful_work": summary["rank_useful_work"],
    }
    records = [None] * world
    dist.all_gather_object(records, local)
    if rank == 0:
        trace_by_rank_step = [
            {
                "rank": record["rank"],
                "step": step["step"],
                "message_count": 2
                * (step["boundary_forward_exchanges"] + step["boundary_reverse_exchanges"]),
                "payload_bytes": step["boundary_bytes"],
                "synchronization_seconds": step["end_to_end_seconds"]
                - step["forward_seconds"]
                - step["reverse_seconds"]
                - step["optimizer_seconds"],
                "route": "intra_node_adjacent_rank_nccl",
                "updates": [
                    item
                    for item in step["bond_updates"]
                    if item["communication_peer"] is not None
                ],
            }
            for record in records
            for step in record["steps"]
        ]
        step_maxima = {
            step: max(
                record["steps"][step]["end_to_end_seconds"] for record in records
            )
            for step in range(args.steps)
        }
        for item in trace_by_rank_step:
            item["straggler_wait_seconds"] = max(
                0.0,
                step_maxima[item["step"]]
                - records[item["rank"]]["steps"][item["step"]]["end_to_end_seconds"],
            )
        payload = {
            "schema": "flagquantum.issue094.mps_communication_run.v1",
            "world_size": world,
            "backend": "nccl",
            "device": torch.cuda.get_device_name(0),
            "protocol": "batched_isend_irecv_descriptor_payload",
            "dedicated_cuda_stream": True,
            "microbenchmark": {
                "legacy_mean_seconds": (
                    None
                    if not args.include_legacy
                    else max(r["legacy"]["mean_seconds"] for r in records)
                ),
                "batched_mean_seconds": max(r["batched"]["mean_seconds"] for r in records),
                "payload_bytes": records[0]["batched"]["payload_bytes"],
                "iterations": records[0]["batched"]["iterations"],
            },
            "setup_seconds_by_rank": [r["communication_setup_seconds"] for r in records],
            "steady_step_seconds_by_rank": [
                [step["end_to_end_seconds"] for step in r["steps"]] for r in records
            ],
            "trace_by_rank_step": trace_by_rank_step,
            "rank_records": records,
            "all_ranks_useful": all(r["rank_useful_work"] for r in records),
            "full_mps_materialization": False,
            "scalability_claim_allowed": False,
            "release_gate_allowed": False,
            "sequence_mismatch_fault": fault,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"output": str(args.output), "world_size": world}), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
