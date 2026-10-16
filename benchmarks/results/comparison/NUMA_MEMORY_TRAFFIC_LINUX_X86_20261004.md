# Linux x86 NUMA memory traffic for differentiable statevectors

## Decision

**Keep the Linux default first-touch policy.** Forced interleave is not an
optimization for the measured native adjoint workloads, and forced bind has
mixed workload/size behavior, so neither is enabled automatically.

## Measured results

`Relative speed` is FQ-default time divided by row time and is above one when
that row is faster. Traffic is uncore IMC CAS read+write bytes per complete
value-and-gradient call.

| Workload | Engine | NUMA policy | Total | DRAM/call | Relative speed | Traffic vs FQ default |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| hardware_efficient_vqe | FlagQuantum native adjoint | default | 61.727 ms | 1.116 GiB | 1.000x | 1.000x |
| hardware_efficient_vqe | FlagQuantum native adjoint | bind | 59.963 ms | 1.105 GiB | 1.029x | 0.990x |
| hardware_efficient_vqe | FlagQuantum native adjoint | interleave | 70.402 ms | 1.512 GiB | 0.877x | 1.355x |
| hardware_efficient_vqe | PennyLane Lightning adjoint | default | 4830.974 ms | 127.181 GiB | 0.013x | 113.946x |
| qaoa_path_maxcut | FlagQuantum native adjoint | default | 55.618 ms | 1.524 GiB | 1.000x | 1.000x |
| qaoa_path_maxcut | FlagQuantum native adjoint | bind | 56.711 ms | 1.569 GiB | 0.981x | 1.029x |
| qaoa_path_maxcut | FlagQuantum native adjoint | interleave | 60.392 ms | 2.027 GiB | 0.921x | 1.330x |
| qaoa_path_maxcut | PennyLane Lightning adjoint | default | 4074.355 ms | 95.644 GiB | 0.014x | 62.743x |

## External same-semantics comparison

- `hardware_efficient_vqe`: FlagQuantum is **78.264x faster** and Lightning
  records **113.946x** as many IMC bytes per call.
- `qaoa_path_maxcut`: FlagQuantum is **73.256x faster** and Lightning records
  **62.743x** as many IMC bytes per call.

## What this measures and why it matters

Each retained call computes one expectation value and its complete parameter
gradient. VQE has independent RX/RY/RZ parameters plus a local Hamiltonian;
QAOA has shared RZZ/RX parameters and a MaxCut Hamiltonian. The result therefore
tests user-visible training evaluations, not an isolated gate microbenchmark.
IMC counters show whether an apparent NUMA speedup merely moves or duplicates
DRAM traffic across sockets.

A minimal public-API call with the same differentiation route is:

```python
import torch
import flagquantum as fq
import flagquantum.algorithms as fqa

theta = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
circuit = fq.Circuit(2, dtype=torch.complex128)
circuit.h(0).cx(0, 1).ry(1, theta)
hamiltonian = fqa.Hamiltonian([fqa.HamiltonianTerm(1.0, "z", 1)])
value = hamiltonian.expectation(circuit, differentiation="adjoint")
gradient = torch.autograd.grad(value, theta)[0]
```

All cells use the same circuit, observable, complex128 precision and
complete gradient semantics. FlagQuantum policy cells use a `1e-09`
correctness tolerance; the independent Lightning reduction uses `1e-06`.
Retained timing rMAD must be at most 20%. The raw samples and per-socket
counters are in
[`numa_memory_traffic_linux_x86_20261004.json`](numa_memory_traffic_linux_x86_20261004.json).

## Reproduce

```bash
export CUDA_VISIBLE_DEVICES=""
export OMP_PROC_BIND=close OMP_PLACES=cores
flagquantum-benchmark run numa_memory_traffic \
  --workloads hardware_efficient_vqe qaoa_path_maxcut \
  --engines flagquantum_adjoint pennylane_lightning_adjoint \
  --policies default bind interleave --nodes 0,1 \
  --cpu-list 0-63 --socket-cpus 0,32 --n-wires 22 --threads 64 \
  --warmup 2 --calls 7 \
  --source-revision 3087092db-plus-feat-numa-memory-traffic-runner \
  --json-output numa_memory_traffic_linux_x86_20261004.json
```

The command needs Linux x86-64, permission to open system-wide perf
events, and Intel IMC PMUs exposing `cas_count_read/write`.

## Boundaries and stopping condition

- CAS counts include unrelated system traffic inside the synchronized window.
- Results cover the listed CPU, 22 qubits, one layer, complex128 and 64 physical
  cores.
- `bind` results do not justify a general runtime default; 20/22/24-qubit
  exploratory probes were also checked before rejecting it.
- This NUMA-policy phase stops when both workloads are correct and stable,
  the native path is observed, an external same-semantics comparison is
  present, and no forced policy improves every measured workload.
