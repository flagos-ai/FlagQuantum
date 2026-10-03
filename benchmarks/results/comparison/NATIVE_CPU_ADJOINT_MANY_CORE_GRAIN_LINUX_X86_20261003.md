# Native CPU adjoint many-core scheduling on Linux x86-64

## Decision

**Accept the adaptive rotation-tile scheduling change.** On a 32-core socket,
the previous 128-item grain could create only 16 tasks for a 22-qubit,
11-wire tile (`2048 / 128`). Asking for 32 worker threads therefore left at
least half of the workers without a tile. The new adaptive grain targets four
chunks per requested worker for every item count instead of becoming fixed at
128 once the item count reaches 1024.

Two independent dual-socket Xeon nodes reproduce the improvement. At 32
threads the complete expectation-value plus full-gradient time improves by
**1.21x for VQE and 1.35x for QAOA** on the primary node, and by **1.19x and
1.35x** on the validation node. All measurements and correctness comparisons
pass. The change does not alter the public API or numerical algorithm.

## What is measured and why it matters

The benchmark executes two representative differentiable quantum workloads:

- hardware-efficient VQE: 22 qubits, 66 independent parameters, 87 gates, and
  a 43-term local Hamiltonian;
- QAOA path MaxCut: 22 qubits, two shared parameters, 65 gates, and a 22-term
  cost Hamiltonian.

The primary metric includes the expectation value and the complete parameter
gradient through FlagQuantum's native statevector adjoint. It answers whether
additional CPU cores reduce the user-visible cost of one training evaluation,
not merely whether one isolated kernel becomes faster.

The rollback engine is the same FlagQuantum implementation with the former
128-item scheduling grain. PennyLane Lightning uses its adjoint method through
the PyTorch interface. Only complete value-and-gradient time is compared
between frameworks because their internal forward/backward accounting differs.

## Environment and method

| Field | Value |
| --- | --- |
| CPU | 2 x Intel Xeon Platinum 8358, 32 physical cores per socket, SMT enabled |
| NUMA | 2 nodes; CPUs 0-31 and 32-63 are the first hardware thread of each socket |
| Memory | 1 TiB |
| OS | Ubuntu 22.04, Linux 5.15, x86-64 |
| Compiler | GCC/G++ 11.4.0, OpenMP 4.5 |
| Python / PyTorch | 3.12.13 / 2.13.0+cu130, CPU device |
| Base source | `e09fd25cf6bf1e2ec7a54879025473c3ca04f2c3` plus this scheduling change |
| Precision | complex128 |
| Sampling | two warmups, seven retained calls, median reported |
| Placement | `OMP_PROC_BIND=close`, `OMP_PLACES=cores`, CUDA hidden |

The timed circuit/device objects are reused; construction, parameter
initialization, gradient zeroing, and optimizer updates are outside the timed
region. Relative median absolute deviation (rMAD) must remain below 20%.

## Matched 32-thread A/B and external comparison

The primary run is pinned to physical cores 0-31 of one NUMA node. Ratios above
one mean the optimized FlagQuantum path is faster.

| Workload | Metric | Optimized | 128-item rollback | Rollback / optimized | PennyLane Lightning adjoint | Lightning / optimized |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| VQE | backward callback | **30.298 ms** | 42.934 ms | **1.417x** | not cross-framework comparable | - |
| VQE | value + full gradient | **64.545 ms** | 78.063 ms | **1.209x** | 3,582.953 ms | **55.511x** |
| QAOA | backward callback | **35.031 ms** | 55.830 ms | **1.594x** | not cross-framework comparable | - |
| QAOA | value + full gradient | **61.511 ms** | 82.860 ms | **1.347x** | 2,776.747 ms | **45.142x** |

All engines pass the `1e-9` value/gradient tolerance. The maximum observed
gradient difference is `3.598e-14`; all value-and-gradient rMAD values are
below 1.6%.

The large Lightning ratios apply only to these exact circuits, versions,
thread settings, and timing boundaries. They are not a universal framework
ranking.

## Independent-node replication

The second node independently fetched the public branch, rebuilt the GCC 11
native extension, passed all 143 native-adjoint unit tests, and repeated the
FlagQuantum A/B:

| Workload | Optimized total | Rollback total | Total speedup | Optimized backward | Rollback backward | Backward speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| VQE | **64.919 ms** | 77.247 ms | **1.190x** | **32.739 ms** | 44.779 ms | **1.368x** |
| QAOA | **60.398 ms** | 81.624 ms | **1.351x** | **37.055 ms** | 57.286 ms | **1.546x** |

Both cells pass correctness and stability; optimized total rMAD is at most
1.61%.

## NUMA and scale boundary

This section characterizes placement rather than claiming linear scalability.

| Placement | Threads | VQE total | QAOA total | Interpretation |
| --- | ---: | ---: | ---: | --- |
| One socket, CPUs 0-31 | 32 | 64.545 ms | 61.511 ms | primary optimized run |
| Split sockets, CPUs 0-15 and 32-47 | 32 | 65.787 ms | 60.283 ms | effectively tied; no reason to span sockets at 32 threads |
| Both sockets, CPUs 0-63 | 64 | 57.511 ms | 50.458 ms | 1.12x/1.22x beyond 32 threads, with diminishing returns |

The 64-core result shows useful additional throughput but not anything close
to 2x. These statevector traversals are increasingly constrained by memory
traffic and cross-socket effects, so this PR does not introduce a NUMA policy
or claim linear thread scaling.

## Raw evidence

- [`linux_x86_adjoint_grain_22q_32t_20261003.json`](linux_x86_adjoint_grain_22q_32t_20261003.json): primary matched A/B plus Lightning
- [`linux_x86_adjoint_grain_22q_32t_node_b_20261003.json`](linux_x86_adjoint_grain_22q_32t_node_b_20261003.json): independent-node A/B
- [`linux_x86_adjoint_grain_22q_32t_split_numa_20261003.json`](linux_x86_adjoint_grain_22q_32t_split_numa_20261003.json): 32 threads split across sockets
- [`linux_x86_adjoint_grain_22q_64t_dual_socket_20261003.json`](linux_x86_adjoint_grain_22q_64t_dual_socket_20261003.json): 64 physical cores across both sockets

Each artifact retains every timing sample, workload hash, correctness error,
runtime version, stability result, and methodology field.

## Reproduce

Build FlagQuantum on Ubuntu 22.04/GCC 11 with the PennyLane extra, then run the
single-socket comparison:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install \
  "torch==2.13.*" --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -e '.[dev,pennylane]'

export CUDA_VISIBLE_DEVICES=""
export OMP_PROC_BIND=close OMP_PLACES=cores

taskset -c 0-31 .venv/bin/python -m \
  flagquantum.benchmarking.differentiable_simulator_corpus \
  --workloads hardware_efficient_vqe qaoa_path_maxcut \
  --n-wires 22 --layers 1 \
  --engines flagquantum_adjoint \
    flagquantum_adjoint_parallel_grain_rollback \
    pennylane_lightning_adjoint \
  --threads 32 --warmup 2 --iterations 7 --calls-per-sample 1 \
  --skip-memory-probe --json-output linux-x86-adjoint-grain.json
```

Use `taskset -c 0-15,32-47` with `--threads 32` for the split-socket placement,
or `taskset -c 0-63` with `--threads 64` for all physical cores.

## Boundaries and stopping condition

- The evidence covers 22-qubit, one-layer VQE/QAOA, complex128, exact
  statevector adjoint on two otherwise identical Linux x86-64 nodes.
- It does not establish universal thread scaling, simultaneous-job throughput,
  SMT benefit, other CPU families, forward-only performance, batch throughput,
  peak memory, or GPU behavior.
- This change is complete because both nodes reproduce the improvement, the
  core numerical suite passes, no measured cell regresses, and the 64-core
  probe identifies memory/NUMA diminishing returns rather than another
  scheduling defect suitable for this PR.
