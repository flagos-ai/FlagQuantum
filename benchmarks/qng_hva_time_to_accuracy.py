"""Matched Adam/reference-QNG/Triton-QNG time-to-accuracy benchmark."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

import flagquantum.algorithms as fqa
from flagquantum.simulation.triton_kernels import heisenberg_hva_forward_tangents


def block_metric(state, tangents, block_size):
    jacobian=tangents.T; connection=torch.conj(state)@jacobian
    full=torch.real(torch.conj(jacobian).T@jacobian-torch.outer(torch.conj(connection),connection));full=0.5*(full+full.T)
    blocked=torch.zeros_like(full)
    for start in range(0,full.shape[0],block_size):blocked[start:start+block_size,start:start+block_size]=full[start:start+block_size,start:start+block_size]
    return blocked


def reference_geometry(n_wires,depth,parameters,block_size):
    def state(value):return fqa.heisenberg_hva(n_wires,depth,value).state().reshape(-1)
    output=state(parameters)
    real=torch.autograd.functional.jacobian(lambda value:state(value).real,parameters,vectorize=True)
    imag=torch.autograd.functional.jacobian(lambda value:state(value).imag,parameters,vectorize=True)
    return block_metric(output,torch.complex(real,imag).T,block_size)


def run(backend,n_wires,depth,steps,seed,lr,damping,wall_budget_seconds=None,compute_exact=True):
    torch.manual_seed(seed);count=fqa.heisenberg_hva_parameter_count(n_wires,depth);block=count//depth
    parameters=(0.02*torch.randn(count,device="cuda")).requires_grad_();hamiltonian=fqa.heisenberg_chain_hamiltonian(n_wires);exact=float(hamiltonian.ground_energy()) if compute_exact else None
    initial=fqa.heisenberg_hva(n_wires,depth,torch.zeros_like(parameters)).state().detach().reshape(-1)
    optimizer=torch.optim.Adam([parameters],lr=lr) if backend=="adam" else None
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();started=time.perf_counter();trace=[]
    def energy():return hamiltonian.expectation(fqa.heisenberg_hva(n_wires,depth,parameters).state()).sum().real
    for step in range(1,steps+1):
        if optimizer is not None:optimizer.zero_grad(set_to_none=True)
        loss=energy();gradient=torch.autograd.grad(loss,parameters)[0]
        if backend=="adam":parameters.grad=gradient;optimizer.step();geometry_seconds=0.0
        else:
            torch.cuda.synchronize();geometry_started=time.perf_counter()
            if backend=="reference_qng":metric=reference_geometry(n_wires,depth,parameters,block)
            else:
                state,tangents=heisenberg_hva_forward_tangents(initial,parameters.detach(),n_wires=n_wires,depth=depth);metric=block_metric(state,tangents,block)
            identity=torch.eye(count,device="cuda");direction=torch.linalg.solve(metric+damping*identity,gradient)
            with torch.no_grad():parameters.add_(-lr*direction)
            torch.cuda.synchronize();geometry_seconds=time.perf_counter()-geometry_started
        recorded=energy();torch.cuda.synchronize();elapsed=time.perf_counter()-started;value=float(recorded.detach())
        trace.append({"optimizer_step":step,"wall_time_seconds":elapsed,"energy":value,"relative_error":None if exact is None else abs(value-exact)/abs(exact),"geometry_seconds":geometry_seconds})
        if wall_budget_seconds is not None and elapsed >= wall_budget_seconds:break
    errors=[p["relative_error"] for p in trace if p["relative_error"] is not None]
    return {"backend":backend,"final_relative_error":trace[-1]["relative_error"],"best_relative_error":min(errors) if errors else None,"time_to_1e-5_seconds":next((p["wall_time_seconds"] for p in trace if p["relative_error"] is not None and p["relative_error"]<=1e-5),None),"completed_steps":len(trace),"peak_memory_bytes":torch.cuda.max_memory_allocated(),"trace":trace}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--n-wires",type=int,default=4);parser.add_argument("--depth",type=int,default=2);parser.add_argument("--steps",type=int,default=60);parser.add_argument("--wall-budget-seconds",type=float,default=None);parser.add_argument("--seed",type=int,default=260720);parser.add_argument("--lr",type=float,default=0.03);parser.add_argument("--damping",type=float,default=1e-3);parser.add_argument("--backends",nargs="+",choices=("adam","reference_qng","triton_qng"),default=("adam","reference_qng","triton_qng"));parser.add_argument("--skip-exact-energy",action="store_true");parser.add_argument("--output",type=Path,required=True);args=parser.parse_args()
    if not torch.cuda.is_available():raise RuntimeError("CUDA required")
    experiments=[run(name,args.n_wires,args.depth,args.steps,args.seed,args.lr,args.damping,args.wall_budget_seconds,not args.skip_exact_energy) for name in args.backends]
    payload={"schema":"flagquantum.qng_hva_time_to_accuracy.v1","execution_semantics":"single_device_fast_path","scalability_claim_allowed":False,"device":torch.cuda.get_device_name(0),"dtype":"complex64","n_wires":args.n_wires,"depth":args.depth,"steps":args.steps,"wall_budget_seconds":args.wall_budget_seconds,"seed":args.seed,"lr":args.lr,"damping":args.damping,"convergence_tolerance":1e-5,"exact_ground_energy":None if args.skip_exact_energy else float(fqa.heisenberg_chain_hamiltonian(args.n_wires).ground_energy()),"experiments":experiments}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8");print(json.dumps({e["backend"]:{"best":e["best_relative_error"],"time_to_1e-5":e["time_to_1e-5_seconds"],"total":e["trace"][-1]["wall_time_seconds"]} for e in experiments},indent=2))


if __name__=="__main__":main()
