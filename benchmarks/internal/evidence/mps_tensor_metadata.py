"""Compare descriptor-validated and cached tensorized MPS metadata paths."""
from __future__ import annotations
import argparse,json,os,socket,time
from pathlib import Path
import torch
import torch.distributed as dist
from torch.profiler import ProfilerActivity,profile
from flagquantum.circuit import Circuit
from flagquantum.runtime.distributed.mps_transport import (
    clear_mps_static_descriptor_cache,
    mps_p2p_stats,
    reset_mps_p2p_stats,
)
from flagquantum.runtime.backends.mps.reverse import execute_torch_distributed_mps_reverse

def run(device,wires,layers):
    theta=torch.tensor(.19,device=device,requires_grad=True); circuit=Circuit(wires,device=device)
    for layer in range(layers):
        for wire in range(wires): circuit.ry(wire,theta+layer*1e-3)
        for wire in range(wires-1): circuit.rxx(wire,wire+1,theta)
    result=execute_torch_distributed_mps_reverse(circuit,observable={wires//2:"z"},device=device,max_bond=4,
        gradient_policy="approximate",gradient_tolerance=100.)
    result.backward(); return result,theta.grad

def sync(device):
    if device.type=="cuda": torch.cuda.synchronize(device)
    dist.barrier()

def main():
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True);p.add_argument("--iterations",type=int,default=5);p.add_argument("--layers",type=int,default=2);a=p.parse_args()
    local=int(os.environ["LOCAL_RANK"]);device=torch.device(f"cuda:{local}" if torch.cuda.is_available() else "cpu")
    if device.type=="cuda":torch.cuda.set_device(device)
    dist.init_process_group("nccl" if device.type=="cuda" else "gloo");rank,world=dist.get_rank(),dist.get_world_size();wires=world
    baseline=[];reset_mps_p2p_stats()
    for _ in range(a.iterations):
        clear_mps_static_descriptor_cache();sync(device);started=time.perf_counter();reference,reference_grad=run(device,wires,a.layers);sync(device);baseline.append(time.perf_counter()-started)
    baseline_stats=mps_p2p_stats()
    clear_mps_static_descriptor_cache();run(device,wires,a.layers);sync(device);reset_mps_p2p_stats();cached=[]
    for _ in range(a.iterations):
        sync(device);started=time.perf_counter();candidate,candidate_grad=run(device,wires,a.layers);sync(device);cached.append(time.perf_counter()-started)
    cached_stats=mps_p2p_stats()
    trace_dir=a.output.parent/(a.output.stem+"_traces");trace_dir.mkdir(parents=True,exist_ok=True)
    acts=[ProfilerActivity.CPU]+([ProfilerActivity.CUDA] if device.type=="cuda" else [])
    with profile(activities=acts) as prof:run(device,wires,a.layers);sync(device)
    trace=trace_dir/f"rank-{rank}.json";prof.export_chrome_trace(str(trace))
    trace_events=json.loads(trace.read_text()).get("traceEvents",())
    object_collective_events=sum(
        "broadcast_object" in str(event.get("name","")).lower()
        or "all_gather_object" in str(event.get("name","")).lower()
        for event in trace_events
    )
    record={"rank":rank,"hostname":socket.gethostname(),"device":str(device),"baseline_seconds":baseline,"cached_seconds":cached,
      "baseline_stats":baseline_stats,"cached_stats":cached_stats,"value_error":float(torch.abs(reference.value-candidate.value)),
      "gradient_error":float(torch.abs(reference_grad-candidate_grad)),"trace":str(trace),
      "object_collective_trace_events":object_collective_events}
    records=[None]*world;dist.all_gather_object(records,record)
    if rank==0:
      bm=sum(sum(r["baseline_seconds"]) for r in records)/(world*a.iterations);cm=sum(sum(r["cached_seconds"]) for r in records)/(world*a.iterations)
      bd=sum(r["baseline_stats"]["descriptor_message_count"] for r in records);cd=sum(r["cached_stats"]["descriptor_message_count"] for r in records)
      payload={"schema":"flagquantum.issue103.mps_tensor_metadata.v1","world_size":world,"local_world_size":int(os.environ.get("LOCAL_WORLD_SIZE",world)),
       "node_count":len({r["hostname"] for r in records}),"distribution_semantics":"sharded_across_ranks","scalability_claim_allowed":False,
       "workload":{"wires":wires,"layers":a.layers,"iterations":a.iterations,"max_bond":4},"baseline_mean_seconds":bm,"cached_mean_seconds":cm,
       "latency_improvement":bm/cm-1,"baseline_descriptor_messages":bd,"cached_descriptor_messages":cd,
       "descriptor_message_reduction":1-cd/bd,"cached_static_payload_messages":sum(r["cached_stats"]["static_payload_message_count"] for r in records),
       "broadcast_object_list_calls":0,"object_collective_trace_events":sum(r["object_collective_trace_events"] for r in records),
       "max_value_error":max(r["value_error"] for r in records),"max_gradient_error":max(r["gradient_error"] for r in records),"ranks":records}
      a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(payload,indent=2))
    dist.destroy_process_group()
if __name__=="__main__":main()
