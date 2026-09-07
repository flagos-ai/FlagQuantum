"""Measure the bounded Z/ZZ objective wavefront for one site-sharded MPS."""
from __future__ import annotations

import argparse
import json
import os
import socket
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.profiler import ProfilerActivity, profile

from flagquantum.runtime.backends.mps.forward import RankOwnedMPSState, _initial_ownership
from flagquantum.runtime.backends.mps.reverse import (
    _fused_z_zz_mse_and_adjoints, _parse_z_zz_terms,
    site_sharded_z_zz_objective_pipeline,
)
from flagquantum.simulation.mps.models import MPSConfig


def states(count: int, wires: int, batch: int, bond: int, device: torch.device):
    rank, world = dist.get_rank(), dist.get_world_size()
    ownership = _initial_ownership(wires, world)
    result = []
    for slot in range(count):
        local = {}
        for wire in ownership[rank]:
            left, right = (1 if wire == 0 else bond), (1 if wire == wires - 1 else bond)
            generator = torch.Generator().manual_seed(9000 + slot * wires + wire)
            # Stable normalized product states embedded in the requested bond
            # shape keep the benchmark finite without reducing kernel shapes.
            angle = torch.randn(batch, generator=generator, dtype=torch.float32)
            tensor = torch.zeros(batch, left, 2, right, dtype=torch.complex64)
            tensor[:, 0, 0, 0] = torch.cos(angle)
            tensor[:, 0, 1, 0] = torch.sin(angle)
            local[wire] = tensor.to(device)
        result.append(RankOwnedMPSState(wires, batch, rank, world, MPSConfig(), local, ownership))
    return tuple(result)


def synchronize(device):
    if device.type == "cuda": torch.cuda.synchronize(device)
    dist.barrier()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--slots", type=int, default=12)
    p.add_argument("--pipeline-slots", type=int, default=8)
    p.add_argument("--wires", type=int, default=128)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--bond", type=int, default=8)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--iterations", type=int, default=5)
    args = p.parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda": torch.cuda.set_device(device)
    dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    values = states(args.slots, args.wires, args.batch, args.bond, device)
    sites = tuple(range(0, args.wires, max(1, args.wires // 16)))
    target = torch.zeros(args.batch, device=device)
    terms = tuple(tuple([({wire: "z"}, target) for wire in sites] +
                        [({wire: "z", wire + 1: "z"}, target) for wire in sites if wire + 1 < args.wires])
                  for _ in values)

    def sequential():
        return tuple(_fused_z_zz_mse_and_adjoints(s, _parse_z_zz_terms(s, t)) for s, t in zip(values, terms))
    def pipelined():
        return site_sharded_z_zz_objective_pipeline(values, terms, max_pipeline_slots=args.pipeline_slots)
    for _ in range(args.warmup): sequential(); pipelined()
    samples = {"sequential": [], "pipeline": []}
    for name, function in (("sequential", sequential), ("pipeline", pipelined)):
        for _ in range(args.iterations):
            synchronize(device); started = time.perf_counter(); result = function(); synchronize(device)
            samples[name].append(time.perf_counter() - started)
    baseline, candidate = sequential(), pipelined()
    value_error = max(float(torch.abs(a[0] - b[0])) for a, b in zip(baseline, candidate))
    grad_error = max(float(torch.max(torch.abs(a[1][wire] - b[1][wire]))) for a, b in zip(baseline, candidate) for wire in a[1])
    update_error = max(float(torch.max(torch.abs((values[i].local_tensors[w] - .01*a[1][w]) - (values[i].local_tensors[w] - .01*b[1][w])))) for i,(a,b) in enumerate(zip(baseline,candidate)) for w in a[1])
    trace_dir = args.output.parent / (args.output.stem + "_traces"); trace_dir.mkdir(parents=True, exist_ok=True)
    activities = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if device.type == "cuda" else [])
    with profile(activities=activities) as prof: pipelined(); synchronize(device)
    trace = trace_dir / f"rank-{rank}.json"; prof.export_chrome_trace(str(trace))
    record = {"rank": rank, "hostname": socket.gethostname(), "device": str(device),
              "owned_wires": values[0].ownership[rank], "sequential_seconds": samples["sequential"],
              "pipeline_seconds": samples["pipeline"], "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None,
              "trace": str(trace), "value_error": value_error, "gradient_error": grad_error, "optimizer_update_error": update_error}
    records = [None] * world; dist.all_gather_object(records, record)
    if rank == 0:
        seq = sum(records[0]["sequential_seconds"]) / args.iterations
        pipe = sum(records[0]["pipeline_seconds"]) / args.iterations
        payload = {"schema": "flagquantum.issue100.mps_pipeline.v1", "world_size": world,
                   "local_world_size": int(os.environ.get("LOCAL_WORLD_SIZE", world)), "node_count": len({r["hostname"] for r in records}),
                   "distribution_semantics": "sharded_across_ranks", "scalability_claim_allowed": False,
                   "slots": args.slots, "max_pipeline_slots": args.pipeline_slots, "partial_final_slots": args.slots % args.pipeline_slots,
                   "sequential_mean_seconds": seq, "pipeline_mean_seconds": pipe,
                   "throughput_improvement": seq / pipe - 1, "batch_one_fast_path": True,
                   "bounded_live_objective_graphs_per_rank": min(args.slots, args.pipeline_slots),
                   "value_error": max(r["value_error"] for r in records), "gradient_error": max(r["gradient_error"] for r in records),
                   "optimizer_update_error": max(r["optimizer_update_error"] for r in records), "ranks": records}
        args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(payload, indent=2))
    dist.destroy_process_group()

if __name__ == "__main__": main()
