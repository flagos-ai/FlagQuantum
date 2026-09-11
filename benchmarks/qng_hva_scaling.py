"""Scale Triton block-QNG against vectorized reverse without training."""
from __future__ import annotations
import argparse,json,statistics,time,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from benchmarks.qng_hva_forward_tangent_triton import metric,reference
import flagquantum.algorithms as fqa
from flagquantum.simulation.triton_kernels import heisenberg_hva_forward_tangents

def measure(fn,warmup,repeats):
 for _ in range(warmup):fn()
 torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();samples=[]
 for _ in range(repeats):
  t=time.perf_counter();fn();torch.cuda.synchronize();samples.append(time.perf_counter()-t)
 return {"samples_seconds":samples,"median_seconds":statistics.median(samples),"peak_memory_bytes":torch.cuda.max_memory_allocated()}

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--n-wires",type=int,required=True);p.add_argument("--depth",type=int,default=2);p.add_argument("--warmup",type=int,default=1);p.add_argument("--repeats",type=int,default=3);p.add_argument("--seed",type=int,default=260720);p.add_argument("--output",type=Path,required=True);a=p.parse_args()
 if not torch.cuda.is_available():raise RuntimeError("CUDA required")
 torch.manual_seed(a.seed);count=fqa.heisenberg_hva_parameter_count(a.n_wires,a.depth);block=count//a.depth;params=(.02*torch.randn(count,device="cuda")).requires_grad_();initial=fqa.heisenberg_hva(a.n_wires,a.depth,torch.zeros_like(params)).state().detach().reshape(-1);h=fqa.heisenberg_chain_hamiltonian(a.n_wires);eye=torch.eye(count,device="cuda")
 def triton_tangent():return heisenberg_hva_forward_tangents(initial,params.detach(),n_wires=a.n_wires,depth=a.depth)
 def reference_tangent():return reference(a.n_wires,a.depth,params)
 def triton_geometry():
  state,tangents=triton_tangent();return metric(state,tangents,block)
 def reference_geometry():
  state,tangents=reference_tangent();return metric(state,tangents,block)
 def objective_gradient():
  energy=h.expectation(fqa.heisenberg_hva(a.n_wires,a.depth,params).state()).sum().real;return torch.autograd.grad(energy,params)[0]
 def triton_direction():return torch.linalg.solve(triton_geometry()+1e-3*eye,objective_gradient())
 def reference_direction():return torch.linalg.solve(reference_geometry()+1e-3*eye,objective_gradient())
 results={}
 for name,fn in (("triton_tangent",triton_tangent),("reference_tangent",reference_tangent),("triton_geometry",triton_geometry),("reference_geometry",reference_geometry),("triton_qng_direction",triton_direction),("reference_qng_direction",reference_direction)):results[name]=measure(fn,a.warmup,a.repeats)
 payload={"schema":"flagquantum.qng_hva_scaling.v1","execution_semantics":"single_device_fast_path","scalability_claim_allowed":False,"device":torch.cuda.get_device_name(0),"dtype":"complex64","n_wires":a.n_wires,"state_dimension":1<<a.n_wires,"depth":a.depth,"parameter_count":count,"warmup":a.warmup,"repeats":a.repeats,"results":results,"speedups":{"tangent":results["reference_tangent"]["median_seconds"]/results["triton_tangent"]["median_seconds"],"geometry":results["reference_geometry"]["median_seconds"]/results["triton_geometry"]["median_seconds"],"qng_direction":results["reference_qng_direction"]["median_seconds"]/results["triton_qng_direction"]["median_seconds"]}}
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n");print(json.dumps({"n":a.n_wires,"parameters":count,"speedups":payload["speedups"]},indent=2))
if __name__=="__main__":main()
