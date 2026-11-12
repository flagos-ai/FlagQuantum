"""Single-device complex128 Torch-FL benchmark with raw synchronized samples."""

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import statistics
import sys
import time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--source', required=True)
parser.add_argument('--qubits', type=int, default=16)
parser.add_argument('--depth', type=int, default=4)
parser.add_argument('--mode', choices=['statevector', 'mps'], default='statevector')
parser.add_argument('--warmup', type=int, default=3)
parser.add_argument('--samples', type=int, default=10)
parser.add_argument('--rounds', type=int, default=3)
parser.add_argument('--label', required=True)
parser.add_argument('--output', required=True)
args = parser.parse_args()

os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD'] = '0'
os.environ['FQ_SV_RUNTIME_MODE'] = os.getenv('FQ_PROBE_RUNTIME_MODE', 'portable')
import torch_fl
import torch

sys.path.insert(0, args.source)
import flagquantum as fq

torch.set_num_threads(2)
torch.set_num_interop_threads(1)
device = torch.device('flagos:0')
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def emit(value):
    print(json.dumps(value, default=str), flush=True)


def sync():
    torch.flagos.synchronize(device)


def circuit(parameters, n, depth):
    c = fq.Circuit(n)
    for layer in range(depth):
        for q in range(n):
            c.ry(q, parameters[layer*n+q])
        for q in range(n-1):
            c.cx(q, q+1)
    return c


def initial(n, depth):
    # Host-defined float64 values are identical across CPU architectures and GPUs.
    return torch.tensor([0.23*math.sin(0.73*(i+1)) + 0.07*math.cos(0.31*(i+1))
                         for i in range(n*depth)], dtype=torch.float64)


def execute(parameters, n, depth, mode, target):
    return fq.run(circuit(parameters, n, depth),
        options=fq.ExecutionOptions(mode=mode, device=target, precision='complex128',
                                    allow_backend_fallback=False),
        outputs=fq.expectation(fq.Z(0)))


def validate(n, depth, mode):
    base = initial(n, depth)
    cpu = base.clone().requires_grad_()
    cpu_result = execute(cpu, n, depth, 'statevector', 'cpu')
    cpu_value = cpu_result.expectation().real.sum()
    cpu_value.backward()
    accelerator = base.to(device).detach().requires_grad_()
    measured = execute(accelerator, n, depth, mode, str(device))
    loss = measured.expectation().real.sum()
    loss.backward()
    sync()
    if loss.device.type != 'flagos' or accelerator.grad.device.type != 'flagos':
        raise RuntimeError('Output or gradient escaped flagos')
    value_error = abs(loss.item()-cpu_value.item())
    grad_error = (accelerator.grad.cpu()-cpu.grad).abs().max().item()
    if value_error > 1e-10 or grad_error > 1e-10:
        raise RuntimeError(f'Numerical mismatch: value={value_error}, gradient={grad_error}')
    return {'qubits': n, 'depth': depth, 'value_error':value_error,
            'gradient_max_error':grad_error, 'output_device':str(loss.device),
            'output_dtype':str(loss.dtype), 'gradient_dtype':str(accelerator.grad.dtype),
            'gradient_device':str(accelerator.grad.device),
            'runtime':dict(measured.runtime), 'provenance':dict(measured.provenance)}


def package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


source_root = Path(args.source)
source_files = sorted(f for folder in ['flagquantum', 'contracts']
                      for f in (source_root/folder).rglob('*')
                      if f.is_file() and '__pycache__' not in f.parts
                      and f.suffix not in {'.pyc', '.pyo'})
source_manifest = {str(f.relative_to(source_root)): hashlib.sha256(f.read_bytes()).hexdigest()
                   for f in source_files}
source_tree_sha256 = hashlib.sha256(json.dumps(source_manifest,sort_keys=True).encode()).hexdigest()
config_path = Path(os.environ['FLAGOS_BACKEND_CONFIG'])
result = {'schema':'flagquantum.cross_vendor.complex128.v1',
    'source_revision':'c2ff8bda879816fc2fa48ce112bfff39b5218764',
    'source_patch':'fused-ry-cx-statevector.patch', 'source_tree_sha256':source_tree_sha256,
    'source_manifest':source_manifest,
    'label':args.label, 'python':sys.version, 'host':platform.node(),
    'torch':torch.__version__, 'torch_fl':package_version('torch_fl'),
    'torch_fl_path':torch_fl.__file__, 'hip':torch.version.hip, 'cuda':torch.version.cuda,
    'cpu':platform.processor(), 'thread_count':torch.get_num_threads(),
    'packages':{p:package_version(p) for p in ['flag_gems','flagtree','triton']},
    'device_name':torch.cuda.get_device_name(0),
    'device_memory_bytes':torch.cuda.get_device_properties(0).total_memory,
    'route_config':str(config_path),
    'route_config_sha256':hashlib.sha256(config_path.read_bytes()).hexdigest(),
    'benchmark_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'mode':args.mode, 'qubits':args.qubits, 'depth':args.depth,
    'parameter_count':args.qubits*args.depth, 'batch_size':1,
    'dtype':'complex128', 'parameter_dtype':'float64', 'optimizer':'SGD(lr=0.01)',
    'gradient_method':'torch_autograd', 'runtime_mode':'portable',
    'distribution_semantics':'single_device_fast_path',
    'world_size':1,'local_world_size':1,'node_count':1,
    'scalability_claim_allowed':False,
    'timing_scope':'Synchronized wall time including public API planning, circuit construction, and execution. Backward-only excludes forward.',
    'warmup':args.warmup,'samples_per_round':args.samples,'rounds':args.rounds,
    'checks':[], 'timings':{}, 'completed':False}


def save():
    Path(args.output).write_text(json.dumps(result,indent=2,default=str)+'\n')


emit({'phase':'environment','label':args.label,'device':result['device_name'],
      'torch':result['torch'],'torch_fl':result['torch_fl'],
      'qubits':args.qubits,'depth':args.depth,'mode':args.mode})
save()
try:
    for n in sorted({min(8,args.qubits),args.qubits}):
        emit({'phase':'correctness_start','qubits':n})
        check=validate(n,args.depth,args.mode)
        result['checks'].append(check)
        emit({'phase':'correctness_pass','qubits':n,'value_error':check['value_error'],
              'gradient_max_error':check['gradient_max_error']})
        save()
    theta=torch.nn.Parameter(initial(args.qubits,args.depth).to(device))
    optimizer=torch.optim.SGD([theta],lr=0.01)
    original=theta.detach().clone()

    def forward():
        with torch.no_grad():
            return execute(theta,args.qubits,args.depth,args.mode,str(device)).expectation().real.sum()

    def loss_with_graph():
        return execute(theta,args.qubits,args.depth,args.mode,str(device)).expectation().real.sum()

    for phase in ['forward','backward_only','forward_backward','training_step']:
        all_samples=[]
        peaks=[]
        round_medians=[]
        for round_index in range(args.rounds):
            with torch.no_grad():
                theta.copy_(original)
            timings=[]
            emit({'phase':'measure','metric':phase,'round':round_index+1})
            for index in range(args.warmup+args.samples):
                optimizer.zero_grad(set_to_none=True)
                loss=loss_with_graph() if phase=='backward_only' else None
                sync()
                torch.cuda.reset_peak_memory_stats()
                started=time.perf_counter_ns()
                if phase=='forward':
                    output=forward()
                elif phase=='backward_only':
                    loss.backward()
                else:
                    output=loss_with_graph()
                    output.backward()
                    if phase=='training_step':
                        optimizer.step()
                sync()
                elapsed=(time.perf_counter_ns()-started)/1e6
                peak=torch.cuda.max_memory_allocated()
                if index>=args.warmup:
                    timings.append(elapsed)
                    peaks.append(peak)
            all_samples.append(timings)
            round_medians.append(statistics.median(timings))
            emit({'phase':'round_complete','metric':phase,'round':round_index+1,
                  'median_ms':round_medians[-1],'peak_allocated_bytes':max(peaks)})
            result['timings'][phase]={'raw_ms':all_samples,'round_medians_ms':round_medians,
                'median_ms':statistics.median(round_medians),'peak_allocated_bytes':max(peaks)}
            save()
    result['completed']=True
    save()
    emit({'phase':'complete','output':args.output,'timings':result['timings']})
except Exception as exc:
    result['error']={'type':type(exc).__name__,'message':str(exc)}
    save()
    raise
