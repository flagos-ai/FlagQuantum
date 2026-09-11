"""End-to-end HVA state/tangent A/B benchmark on A800."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

import flagquantum.algorithms as fqa
from flagquantum.simulation.triton_kernels import heisenberg_hva_forward_tangents


def metric(state, tangents, block_size):
    jacobian = tangents.T
    connection = torch.conj(state) @ jacobian
    full = torch.real(torch.conj(jacobian).T @ jacobian - torch.outer(torch.conj(connection), connection))
    full = 0.5 * (full + full.T)
    blocked = torch.zeros_like(full)
    for start in range(0, full.shape[0], block_size):
        blocked[start:start + block_size, start:start + block_size] = full[start:start + block_size, start:start + block_size]
    return blocked


def reference(n_wires, depth, parameters):
    def state(value):
        return fqa.heisenberg_hva(n_wires, depth, value).state().reshape(-1)
    output = state(parameters)
    real = torch.autograd.functional.jacobian(lambda value: state(value).real, parameters, vectorize=True)
    imag = torch.autograd.functional.jacobian(lambda value: state(value).imag, parameters, vectorize=True)
    return output, torch.complex(real, imag).T


def measure(function, warmup, repeats):
    for _ in range(warmup): function()
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    samples=[]
    for _ in range(repeats):
        started=time.perf_counter(); function(); torch.cuda.synchronize(); samples.append(time.perf_counter()-started)
    return {"samples_seconds":samples,"median_seconds":statistics.median(samples),"peak_memory_bytes":torch.cuda.max_memory_allocated()}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-wires",type=int,default=6); parser.add_argument("--depth",type=int,default=3)
    parser.add_argument("--warmup",type=int,default=2); parser.add_argument("--repeats",type=int,default=5)
    parser.add_argument("--seed",type=int,default=260720); parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError("CUDA is required")
    torch.manual_seed(args.seed)
    count=fqa.heisenberg_hva_parameter_count(args.n_wires,args.depth); block=count//args.depth
    parameters=0.02*torch.randn(count,device="cuda")
    initial=fqa.heisenberg_hva(args.n_wires,args.depth,torch.zeros_like(parameters)).state().detach().reshape(-1)
    triton_state,triton_tangents=heisenberg_hva_forward_tangents(initial,parameters,n_wires=args.n_wires,depth=args.depth)
    reference_state,reference_tangents=reference(args.n_wires,args.depth,parameters)
    triton_metric=metric(triton_state,triton_tangents,block); reference_metric=metric(reference_state,reference_tangents,block)
    gradient=torch.randn(count,device="cuda",generator=torch.Generator(device="cuda").manual_seed(args.seed+1)); eye=torch.eye(count,device="cuda")
    td=torch.linalg.solve(triton_metric+1e-3*eye,gradient); rd=torch.linalg.solve(reference_metric+1e-3*eye,gradient)
    triton_result=measure(lambda:heisenberg_hva_forward_tangents(initial,parameters,n_wires=args.n_wires,depth=args.depth),args.warmup,args.repeats)
    reference_result=measure(lambda:reference(args.n_wires,args.depth,parameters),args.warmup,args.repeats)
    payload={
      "schema":"flagquantum.qng_hva_forward_tangent_triton.v1","execution_semantics":"single_device_fast_path","scalability_claim_allowed":False,
      "device":torch.cuda.get_device_name(0),"dtype":"complex64","n_wires":args.n_wires,"depth":args.depth,"parameter_count":count,
      "maximum_state_error":float(torch.max(torch.abs(triton_state-reference_state))),
      "relative_l2_tangent_error":float(torch.linalg.vector_norm(triton_tangents-reference_tangents)/torch.linalg.vector_norm(reference_tangents)),
      "relative_l2_metric_error":float(torch.linalg.vector_norm(triton_metric-reference_metric)/torch.linalg.vector_norm(reference_metric)),
      "qng_direction_cosine_similarity":float(torch.dot(td,rd)/(torch.linalg.vector_norm(td)*torch.linalg.vector_norm(rd))),
      "triton":triton_result,"vectorized_reverse":reference_result,
      "speedup_vectorized_reverse_over_triton":reference_result["median_seconds"]/triton_result["median_seconds"]}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8");print(json.dumps(payload,indent=2))


if __name__=="__main__":main()
