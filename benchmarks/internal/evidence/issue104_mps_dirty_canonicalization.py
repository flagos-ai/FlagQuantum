"""Compare full-chain and dirty-interval MPS canonicalization."""
from __future__ import annotations
import argparse,json,os,socket,time
from pathlib import Path
import torch
import torch.distributed as dist
from torch.profiler import ProfilerActivity,profile
from flagquantum.circuit import Circuit
from flagquantum.runtime.backends.mps.reverse import execute_torch_distributed_mps_reverse

def circuit(theta,wires,two_site=True):
    c=Circuit(wires,device=theta.device);c.ry(wires-2,theta)
    if two_site:c.rxx(wires-2,wires-1,theta)
    c.ry(wires-1,theta);return c

def run(device,wires,policy,two_site=True,max_bond=None,gradient_policy="exact"):
    theta=torch.tensor(.27,device=device,requires_grad=True)
    result=execute_torch_distributed_mps_reverse(circuit(theta,wires,two_site),observable={wires-1:"z"},device=device,
      max_bond=max_bond,canonicalization_policy=policy,gradient_policy=gradient_policy,
      gradient_tolerance=10. if gradient_policy=="approximate" else 0.)
    result.backward();return result,theta.grad

def sync(device):
    if device.type=="cuda":torch.cuda.synchronize(device)
    dist.barrier()

def measured(device,wires,policy,iters):
    samples=[];last=None;grad=None
    for _ in range(iters):sync(device);start=time.perf_counter();last,grad=run(device,wires,policy);sync(device);samples.append(time.perf_counter()-start)
    return last,grad,samples

def main():
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True);p.add_argument("--wires",type=int,default=64);p.add_argument("--iterations",type=int,default=5);a=p.parse_args()
    local=int(os.environ["LOCAL_RANK"]);device=torch.device(f"cuda:{local}" if torch.cuda.is_available() else "cpu")
    if device.type=="cuda":torch.cuda.set_device(device)
    dist.init_process_group("nccl" if device.type=="cuda" else "gloo");rank,world=dist.get_rank(),dist.get_world_size()
    run(device,a.wires,"full");run(device,a.wires,"dirty")
    full,fg,full_samples=measured(device,a.wires,"full",a.iterations);dirty,dg,dirty_samples=measured(device,a.wires,"dirty",a.iterations)
    clean,_=run(device,a.wires,"dirty",two_site=False)
    approximate,ag=run(device,a.wires,"dirty",max_bond=1,gradient_policy="approximate")
    trace_dir=a.output.parent/(a.output.stem+"_traces");trace_dir.mkdir(parents=True,exist_ok=True)
    acts=[ProfilerActivity.CPU]+([ProfilerActivity.CUDA] if device.type=="cuda" else [])
    trace_records={}
    for name,policy in (("full","full"),("dirty","dirty")):
      with profile(activities=acts) as prof:run(device,a.wires,policy);sync(device)
      path=trace_dir/f"rank-{rank}.{name}.json";prof.export_chrome_trace(str(path));events=json.loads(path.read_text()).get("traceEvents",())
      factors=[e for e in events if e.get("ph")=="X" and e.get("name") in ("flagquantum::mps::qr","flagquantum::mps::svd")]
      trace_records[name]={"path":str(path),"factorization_event_count":len(factors),"factorization_time_us":sum(float(e.get("dur",0)) for e in factors)}
    rec={"rank":rank,"hostname":socket.gethostname(),"device":str(device),"full_seconds":full_samples,"dirty_seconds":dirty_samples,
      "full_summary":full.summary(),"dirty_summary":dirty.summary(),"clean_summary":clean.summary(),"approximate_summary":approximate.summary(),
      "value_error":float(torch.abs(full.value-dirty.value)),"gradient_error":float(torch.abs(fg-dg)),"approximate_gradient_finite":bool(torch.isfinite(ag)),
      "exact_discarded_weight":dirty.discarded_weight,"approximate_discarded_weight":approximate.discarded_weight,
      "approximate_tolerance":approximate.gradient_tolerance,
      "minimum_truncation_gap":min((r.singular_value_gap for r in approximate.tape.records if r.singular_value_gap is not None),default=None),
      "traces":trace_records}
    records=[None]*world;dist.all_gather_object(records,rec)
    if rank==0:
      fm=sum(sum(r["full_seconds"]) for r in records)/(world*a.iterations);dm=sum(sum(r["dirty_seconds"]) for r in records)/(world*a.iterations)
      ft=sum(r["traces"]["full"]["factorization_time_us"] for r in records);dt=sum(r["traces"]["dirty"]["factorization_time_us"] for r in records)
      payload={"schema":"flagquantum.issue104.mps_dirty_canonicalization.v1","world_size":world,"local_world_size":int(os.environ.get("LOCAL_WORLD_SIZE",world)),
       "node_count":len({r["hostname"] for r in records}),"distribution_semantics":"sharded_across_ranks","scalability_claim_allowed":False,
       "workload":{"wires":a.wires,"sparse_gate_bonds":[a.wires-2],"iterations":a.iterations},"full_mean_seconds":fm,"dirty_mean_seconds":dm,
       "step_latency_improvement":fm/dm-1,"full_factorization_time_us":ft,"dirty_factorization_time_us":dt,"factorization_time_reduction":1-dt/ft,
       "full_planned_qr_bonds":len(full.planned_canonicalization_bonds),"dirty_planned_qr_bonds":len(dirty.planned_canonicalization_bonds),
       "certified_clean_planned_qr_bonds":len(clean.planned_canonicalization_bonds),"max_value_error":max(r["value_error"] for r in records),
       "max_gradient_error":max(r["gradient_error"] for r in records),"exact_discarded_weight":max(r["exact_discarded_weight"] for r in records),
       "approximate_discarded_weight":max(r["approximate_discarded_weight"] for r in records),"approximate_tolerance":records[0]["approximate_tolerance"],
       "minimum_truncation_gap":min(r["minimum_truncation_gap"] for r in records if r["minimum_truncation_gap"] is not None),
       "approximate_gradients_finite":all(r["approximate_gradient_finite"] for r in records),"ranks":records}
      a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(payload,indent=2))
    dist.destroy_process_group()
if __name__=="__main__":main()
