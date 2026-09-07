"""Compare record-wise and fused owner-local MPS reverse dispatch."""
from __future__ import annotations

import argparse
import json
import os
import socket
import time
from pathlib import Path

import torch
import torch.distributed as dist

from flagquantum.circuit import Circuit
from flagquantum.runtime.backends.mps.forward import _initial_ownership
from flagquantum.runtime.backends.mps.reverse import execute_torch_distributed_mps_reverse
from flagquantum.simulation.mps.site_kernels import (
    reset_site_kernel_stats,
    site_kernel_stats,
)


def make_case(theta, wires, layers, device):
    circuit = Circuit(wires, device=device)
    for layer in range(layers):
        for wire in range(wires):
            circuit.ry(wire, theta + (layer + wire) * 1e-4)
    rank, world = dist.get_rank(), dist.get_world_size()
    ownership = _initial_ownership(wires, world)
    initial = {}
    for wire in ownership[rank]:
        tensor = torch.zeros(1, 1, 2, 1, dtype=torch.complex64, device=device)
        tensor[:, :, 0, :] = 1
        initial[wire] = tensor
    return circuit, initial


def synchronize(device):
    if device.type == "cuda": torch.cuda.synchronize(device)
    dist.barrier()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wires", type=int, default=32)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=3)
    args = parser.parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda": torch.cuda.set_device(device)
    dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    reset_site_kernel_stats()

    def run(fused):
        theta = torch.tensor(0.23, device=device, requires_grad=True)
        circuit, initial = make_case(theta, args.wires, args.layers, device)
        result = execute_torch_distributed_mps_reverse(
            circuit, observable={args.wires // 2: "z"}, device=device,
            initial_mps_tensors=initial, initial_mps_left_canonical=True,
            compile_site_kernels=fused, fuse_local_reverse=fused,
        )
        if device.type == "cuda": torch.cuda.reset_peak_memory_stats(device)
        synchronize(device); started = time.perf_counter(); result.backward(); synchronize(device)
        seconds = time.perf_counter() - started
        peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
        return result, theta.grad.detach(), seconds, int(peak)

    for _ in range(args.warmup): run(False); run(True)
    samples = {"record": [], "fused": []}; memories = {"record": [], "fused": []}
    reference_grad = candidate_grad = None; reference = candidate = None
    for name, fused in (("record", False), ("fused", True)):
        for _ in range(args.iterations):
            result, grad, seconds, peak = run(fused)
            samples[name].append(seconds); memories[name].append(peak)
            if fused: candidate, candidate_grad = result, grad
            else: reference, reference_grad = result, grad
    record = {"rank": rank, "hostname": socket.gethostname(), "device": str(device),
              "record_seconds": samples["record"], "fused_seconds": samples["fused"],
              "record_peak_live_bytes": max(memories["record"]), "fused_peak_live_bytes": max(memories["fused"]),
              "gradient_error": float(torch.abs(reference_grad-candidate_grad)),
              "record_summary": reference.summary(), "fused_summary": candidate.summary(),
              "compiled_kernel_stats": site_kernel_stats()}
    records = [None] * world; dist.all_gather_object(records, record)
    if rank == 0:
        record_mean = sum(record["record_seconds"])/args.iterations
        fused_mean = sum(record["fused_seconds"])/args.iterations
        baseline_calls = sum(r["record_summary"]["reverse_autograd_grad_invocations"] for r in records)
        fused_calls = sum(r["fused_summary"]["reverse_autograd_grad_invocations"] for r in records)
        baseline_dispatch = records[0]["record_summary"]["reverse_python_dispatches"]
        fused_dispatch = records[0]["fused_summary"]["reverse_python_dispatches"]
        baseline_memory = max(r["record_peak_live_bytes"] for r in records)
        fused_memory = max(r["fused_peak_live_bytes"] for r in records)
        payload = {"schema":"flagquantum.issue101.mps_fused_reverse.v1", "world_size":world,
          "local_world_size":int(os.environ.get("LOCAL_WORLD_SIZE",world)), "node_count":len({r["hostname"] for r in records}),
          "distribution_semantics":"sharded_across_ranks", "scalability_claim_allowed":False,
          "workload":{"wires":args.wires,"layers":args.layers,"batch":1,"initial_mps_left_canonical":True},
          "record_reverse_mean_seconds":record_mean,"fused_reverse_mean_seconds":fused_mean,
          "reverse_time_improvement":record_mean/fused_mean-1,
          "baseline_autograd_grad_invocations":baseline_calls,"fused_autograd_grad_invocations":fused_calls,
          "autograd_invocation_reduction":1-fused_calls/baseline_calls,
          "baseline_python_dispatches":baseline_dispatch,"fused_python_dispatches":fused_dispatch,
          "python_dispatch_reduction":1-fused_dispatch/baseline_dispatch,
          "baseline_peak_live_bytes":baseline_memory,"fused_peak_live_bytes":fused_memory,
          "peak_live_memory_growth":(fused_memory/baseline_memory-1) if baseline_memory else 0,
          "max_gradient_error":max(r["gradient_error"] for r in records), "ranks":records}
        payload["compiled_graph_count"] = sum(r["compiled_kernel_stats"]["dynamo_graphs"] for r in records)
        payload["generated_kernel_count"] = sum(r["compiled_kernel_stats"]["triton_kernels"] for r in records)
        args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(payload,indent=2))
    dist.destroy_process_group()

if __name__ == "__main__": main()
