"""Audit owner gradient buckets with one thousand MPS parameters."""
from __future__ import annotations

import argparse, json, os, socket, time
from pathlib import Path
import torch
import torch.distributed as dist
from torch.profiler import ProfilerActivity, profile

from flagquantum.circuit import Circuit
from flagquantum.runtime.backends.mps.forward import _initial_ownership
from flagquantum.runtime.backends.mps.reverse import execute_torch_distributed_mps_reverse


def build(count, wires, device):
    values = tuple(torch.tensor(.001 * (index % 17), device=device, requires_grad=True) for index in range(count))
    circuit = Circuit(wires, device=device)
    for index, value in enumerate(values): circuit.ry(index % wires, value)
    ownership = _initial_ownership(wires, dist.get_world_size())
    initial = {}
    for wire in ownership[dist.get_rank()]:
        tensor=torch.zeros(1,1,2,1,dtype=torch.complex64,device=device); tensor[:,:,0,:]=1; initial[wire]=tensor
    return circuit, values, initial


def execute(count,wires,device,owners,bucket_bytes):
    circuit, values, initial = build(count,wires,device)
    result=execute_torch_distributed_mps_reverse(circuit,observable={0:"z"},device=device,
        initial_mps_tensors=initial,initial_mps_left_canonical=True,fuse_local_reverse=False,
        gradient_owner_ranks=owners,gradient_bucket_bytes=bucket_bytes)
    if device.type=="cuda": torch.cuda.synchronize(device)
    started=time.perf_counter(); result.backward()
    if device.type=="cuda": torch.cuda.synchronize(device)
    return result,values,time.perf_counter()-started


def optimizer_parity(reference,candidate,owners,rank,kind):
    ref=[torch.nn.Parameter(p.detach().clone()) for p in reference]
    got=[torch.nn.Parameter(p.detach().clone()) for p in candidate]
    owned=[i for i,o in enumerate(owners) if o==rank]
    for i in owned: ref[i].grad=reference[i].grad.detach().clone(); got[i].grad=candidate[i].grad.detach().clone()
    cls=torch.optim.Adam if kind=="adam" else torch.optim.SGD
    a=cls([ref[i] for i in owned],lr=.01); b=cls([got[i] for i in owned],lr=.01)
    a.step(); b.step()
    state=b.state_dict(); resumed=cls([got[i] for i in owned],lr=.01); resumed.load_state_dict(state)
    for i in owned: ref[i].grad=reference[i].grad; got[i].grad=candidate[i].grad
    a.step(); resumed.step()
    error=max((float(torch.abs(ref[i].detach()-got[i].detach())) for i in owned),default=0.)
    return error, len(resumed.state)


def main():
    p=argparse.ArgumentParser(); p.add_argument("--output",type=Path,required=True); p.add_argument("--parameters",type=int,default=1000); args=p.parse_args()
    local=int(os.environ["LOCAL_RANK"]); device=torch.device(f"cuda:{local}" if torch.cuda.is_available() else "cpu")
    if device.type=="cuda": torch.cuda.set_device(device)
    dist.init_process_group("nccl" if device.type=="cuda" else "gloo"); rank,world=dist.get_rank(),dist.get_world_size(); wires=world
    owners=tuple(i%world for i in range(args.parameters))
    baseline,ref,base_seconds=execute(args.parameters,wires,device,None,4)
    candidate,got,bucket_seconds=execute(args.parameters,wires,device,owners,1<<20)
    gradient_error=max((float(torch.abs(ref[i].grad-got[i].grad)) for i,o in enumerate(owners) if o==rank),default=0.)
    nonowner_materialized=sum(got[i].grad is not None for i,o in enumerate(owners) if o!=rank)
    sgd_error,sgd_state=optimizer_parity(ref,got,owners,rank,"sgd")
    adam_error,adam_state=optimizer_parity(ref,got,owners,rank,"adam")
    trace_dir=args.output.parent/(args.output.stem+"_traces"); trace_dir.mkdir(parents=True,exist_ok=True)
    activities=[ProfilerActivity.CPU]+([ProfilerActivity.CUDA] if device.type=="cuda" else [])
    with profile(activities=activities) as prof:
        traced,traced_values,_=execute(args.parameters,wires,device,owners,1<<20)
    trace=trace_dir/f"rank-{rank}.json"; prof.export_chrome_trace(str(trace)); del traced,traced_values
    record={"rank":rank,"hostname":socket.gethostname(),"device":str(device),"gradient_error":gradient_error,
      "nonowner_materialized_gradients":nonowner_materialized,"sgd_update_error":sgd_error,"adam_update_error":adam_error,
      "sgd_state_entries":sgd_state,"adam_state_entries":adam_state,"baseline_seconds":base_seconds,"bucket_seconds":bucket_seconds,
      "baseline_summary":baseline.summary(),"bucket_summary":candidate.summary(),"trace":str(trace)}
    records=[None]*world; dist.all_gather_object(records,record)
    if rank==0:
      bc=records[0]["baseline_summary"]["gradient_collective_count"]; cc=records[0]["bucket_summary"]["gradient_collective_count"]
      payload={"schema":"flagquantum.issue102.mps_gradient_buckets.v1","world_size":world,"local_world_size":int(os.environ.get("LOCAL_WORLD_SIZE",world)),
       "node_count":len({r["hostname"] for r in records}),"distribution_semantics":"sharded_across_ranks","scalability_claim_allowed":False,
       "parameter_count":args.parameters,"owner_bucket_count":world,"baseline_collective_count":bc,"bucket_collective_count":cc,
       "collective_count_reduction":1-cc/bc,"max_gradient_error":max(r["gradient_error"] for r in records),
       "max_sgd_update_error":max(r["sgd_update_error"] for r in records),"max_adam_update_error":max(r["adam_update_error"] for r in records),
       "nonowner_materialized_gradients":sum(r["nonowner_materialized_gradients"] for r in records),
       "optimizer_resume_state_entries":sum(r["adam_state_entries"] for r in records),"loss_error":float(torch.abs(baseline.value-candidate.value)),
       "baseline_mean_rank_seconds":sum(r["baseline_seconds"] for r in records)/world,"bucket_mean_rank_seconds":sum(r["bucket_seconds"] for r in records)/world,
       "ranks":records}
      args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(payload,indent=2))
    dist.destroy_process_group()
if __name__=="__main__": main()
