# Linux x86-64 CPU simulator scorecard

## Decision

**Accept this artifact as the formal Linux x86-64 baseline, not as a claim that
FlagQuantum is the fastest simulator in every regime.** On one 32-core Xeon
socket, FlagQuantum is faster than the standard PennyLane `lightning.qubit`
wheel in all six 22-qubit forward workloads, but Qiskit Aer remains faster in
four of those six workloads. FlagQuantum is the fastest measured engine for
22-qubit Dense nonlocal and SWAP routing. The accepted adjoint evidence remains
strong: FlagQuantum is 45.14x-55.51x faster than PennyLane Lightning adjoint at
22 qubits and 32 threads for the two measured training workloads.

The forward result identifies the next engineering target without guessing:
socket-local multi-task throughput comes first, followed by NUMA traffic
measurement and then portable x86 kernels for the largest reproducible gaps.
The baseline itself changes no simulator behavior or public API.

## What is measured and why it matters

The forward corpus covers six distinct exact-statevector structures:

- Hardware-efficient: interleaved parameterized one-qubit rotations and a CX
  chain;
- Truncated QFT: Hadamards, bounded controlled phases, and terminal swaps;
- Random Clifford: deterministic Clifford-dominated gates;
- Local brickwork: repeated nearest-neighbor two-qubit layers;
- Dense nonlocal: wide nonlocal two-qubit connectivity;
- SWAP routing: permutation-heavy state movement.

Each engine receives the same FlagQuantum circuit and returns the complete
complex128 statevector. The timed region is the user-facing run call, including
conversion, backend/device preparation, execution, and result retrieval. This
is deliberately not an isolated-kernel contest.

The adjoint rows measure expectation value plus the complete parameter
gradient for VQE and QAOA. They are linked here rather than rerun because the
accepted artifacts already use these exact hosts, compiler, precision, sample
count, and thread placements.

## Environment and method

| Field | Value |
| --- | --- |
| Primary host | `jp-a800-172` |
| CPU | 2 x Intel Xeon Platinum 8358, 32 physical cores per socket, SMT enabled |
| NUMA | node 0: CPUs 0-31,64-95; node 1: CPUs 32-63,96-127 |
| Memory | 1 TiB |
| OS | Ubuntu 22.04, Linux 5.15, x86-64 |
| Compiler | GCC/G++ 11.4.0 |
| Python / PyTorch | 3.12.13 / 2.13.0+cu130, CPU device |
| FlagQuantum source | `8ecc1965adb81b6b8dfeb39e46211efee75b2b24` |
| External engines | Qiskit Aer 0.17.2; PennyLane 0.45.1 / Lightning 0.45.0 |
| Precision | complex128 |
| Sampling | two warmups, seven retained calls, median reported |
| Placement | one thread on CPU 32; 32 threads on physical CPUs 32-63 |
| Thread policy | `OMP_PROC_BIND=close`, `OMP_PLACES=cores`, CUDA hidden |

Every one of the 24 forward case/thread cells passes exact-statevector
correctness at the corpus tolerance and the 20% relative-MAD stability rule.
Engines rotate their execution order within one process.

## Complete forward comparison

`Best external / FQ` is the faster of Qiskit Aer and PennyLane Lightning
divided by FlagQuantum. A ratio above one means FlagQuantum is faster; a ratio
below one means an external engine is faster.

| Workload | Qubits | Threads | FlagQuantum | PennyLane Lightning | Qiskit Aer | Best external / FQ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Hardware-efficient | 18 | 1 | 31.185 ms | 30.306 ms | 128.407 ms | **0.972x** |
| Hardware-efficient | 18 | 32 | 15.615 ms | 30.099 ms | 71.281 ms | **1.928x** |
| Hardware-efficient | 22 | 1 | 1,081.622 ms | 900.984 ms | 1,573.845 ms | **0.833x** |
| Hardware-efficient | 22 | 32 | 225.394 ms | 700.389 ms | 135.207 ms | **0.600x** |
| Truncated QFT | 18 | 1 | 15.583 ms | 52.174 ms | 137.408 ms | **3.348x** |
| Truncated QFT | 18 | 32 | 14.109 ms | 52.775 ms | 75.675 ms | **3.741x** |
| Truncated QFT | 22 | 1 | 687.326 ms | 1,348.121 ms | 1,708.210 ms | **1.961x** |
| Truncated QFT | 22 | 32 | 235.890 ms | 1,214.162 ms | 146.023 ms | **0.619x** |
| Random Clifford | 18 | 1 | 30.114 ms | 20.893 ms | 115.373 ms | **0.694x** |
| Random Clifford | 18 | 32 | 22.493 ms | 20.559 ms | 68.494 ms | **0.914x** |
| Random Clifford | 22 | 1 | 1,065.613 ms | 614.464 ms | 1,397.505 ms | **0.577x** |
| Random Clifford | 22 | 32 | 294.427 ms | 471.661 ms | 129.035 ms | **0.438x** |
| Local brickwork | 18 | 1 | 37.411 ms | 36.173 ms | 123.443 ms | **0.967x** |
| Local brickwork | 18 | 32 | 16.816 ms | 36.246 ms | 72.281 ms | **2.155x** |
| Local brickwork | 22 | 1 | 1,795.069 ms | 934.132 ms | 1,491.146 ms | **0.520x** |
| Local brickwork | 22 | 32 | 407.109 ms | 870.358 ms | 143.307 ms | **0.352x** |
| Dense nonlocal | 18 | 1 | 19.724 ms | 31.118 ms | 201.741 ms | **1.578x** |
| Dense nonlocal | 18 | 32 | 8.470 ms | 31.236 ms | 80.070 ms | **3.688x** |
| Dense nonlocal | 22 | 1 | 855.083 ms | 585.914 ms | 3,453.167 ms | **0.685x** |
| Dense nonlocal | 22 | 32 | 122.395 ms | 541.152 ms | 216.061 ms | **1.765x** |
| SWAP routing | 18 | 1 | 12.485 ms | 24.044 ms | 91.847 ms | **1.926x** |
| SWAP routing | 18 | 32 | 11.646 ms | 24.027 ms | 66.034 ms | **2.063x** |
| SWAP routing | 22 | 1 | 47.327 ms | 625.796 ms | 888.306 ms | **13.223x** |
| SWAP routing | 22 | 32 | 26.105 ms | 587.669 ms | 110.278 ms | **4.224x** |

## Socket-local thread scaling

The 32-thread speedup is relative to the pinned single-thread result on the
same NUMA socket. FlagQuantum gains 1.07x-6.99x, not 32x. This is expected for
short 18-qubit tasks and memory-bound full-state traversals, but the four
22-qubit cases where Aer scales by more than 10x identify a concrete remaining
parallelism gap rather than a reason to claim linear scaling.

| Workload | Qubits | FlagQuantum 1 thread | FlagQuantum 32 threads | FQ speedup | Lightning speedup | Aer speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Hardware-efficient | 18 | 31.185 ms | 15.615 ms | **1.997x** | 1.007x | 1.801x |
| Hardware-efficient | 22 | 1,081.622 ms | 225.394 ms | **4.799x** | 1.286x | 11.640x |
| Truncated QFT | 18 | 15.583 ms | 14.109 ms | **1.105x** | 0.989x | 1.816x |
| Truncated QFT | 22 | 687.326 ms | 235.890 ms | **2.914x** | 1.110x | 11.698x |
| Random Clifford | 18 | 30.114 ms | 22.493 ms | **1.339x** | 1.016x | 1.684x |
| Random Clifford | 22 | 1,065.613 ms | 294.427 ms | **3.619x** | 1.303x | 10.830x |
| Local brickwork | 18 | 37.411 ms | 16.816 ms | **2.225x** | 0.998x | 1.708x |
| Local brickwork | 22 | 1,795.069 ms | 407.109 ms | **4.409x** | 1.073x | 10.405x |
| Dense nonlocal | 18 | 19.724 ms | 8.470 ms | **2.329x** | 0.996x | 2.520x |
| Dense nonlocal | 22 | 855.083 ms | 122.395 ms | **6.986x** | 1.083x | 15.982x |
| SWAP routing | 18 | 12.485 ms | 11.646 ms | **1.072x** | 1.001x | 1.391x |
| SWAP routing | 22 | 47.327 ms | 26.105 ms | **1.813x** | 1.065x | 8.055x |

## Differentiable adjoint baseline

The accepted Linux artifacts provide the matched adjoint baseline. The
32-thread rows below use one physical socket on `jp-a800-171`; an independent
run on `jp-a800-172` reproduced the FlagQuantum improvement.

| Workload | Qubits | Threads | FlagQuantum value + gradient | PennyLane Lightning adjoint | Lightning / FQ |
| --- | ---: | ---: | ---: | ---: | ---: |
| Hardware-efficient VQE | 18 | 1 | 42.525 ms | 190.456 ms | **4.479x** |
| QAOA path MaxCut | 18 | 1 | 49.222 ms | 128.460 ms | **2.610x** |
| Hardware-efficient VQE | 22 | 1 | 760.967 ms | 4,425.672 ms | **5.816x** |
| QAOA path MaxCut | 22 | 1 | 942.468 ms | 2,800.858 ms | **2.972x** |
| Hardware-efficient VQE | 22 | 32 | 64.545 ms | 3,582.953 ms | **55.511x** |
| QAOA path MaxCut | 22 | 32 | 61.511 ms | 2,776.747 ms | **45.142x** |

See
[`LINUX_X86_GCC11_NATIVE_CPU_ADJOINT_20261003.md`](LINUX_X86_GCC11_NATIVE_CPU_ADJOINT_20261003.md)
and
[`NATIVE_CPU_ADJOINT_MANY_CORE_GRAIN_LINUX_X86_20261003.md`](NATIVE_CPU_ADJOINT_MANY_CORE_GRAIN_LINUX_X86_20261003.md)
for every retained sample, correctness error, rollback A/B, and the second-node
replication.

## What about Lightning-Kokkos?

The installed PennyLane 0.45.1 / Lightning 0.45.0 Linux wheel registers
`lightning.qubit` but not `lightning.kokkos`. A Kokkos result is therefore
**not available**, rather than zero or slower. Producing one requires a separate
source build with its compiler and OpenMP settings recorded. Kokkos remains a
candidate portability experiment, not a FlagQuantum dependency or a result in
this table. A later prototype is justified only if it beats FlagQuantum by at
least 1.25x in multiple important same-scope workloads or materially reduces
backend maintenance without regressing correctness.

Reproduce the installed-device boundary with:

```bash
.venv/bin/python -c \
  'import pennylane as qml; print(sorted(qml.plugin_devices))'
```

## Public example

The measured native path is the ordinary public statevector API; users do not
select a benchmark-only backend:

```python
import torch
import flagquantum as fq

circuit = fq.Circuit(22, dtype=torch.complex128)
for wire in range(22):
    circuit.h(wire)
for wire in range(21):
    circuit.cx(wire, wire + 1)

state = circuit.state(refresh=True)
assert state.shape == (2**22,)
```

## Raw evidence and reproduction

- [`linux_x86_cpu_scorecard_forward_1t_20261003.json`](linux_x86_cpu_scorecard_forward_1t_20261003.json)
- [`linux_x86_cpu_scorecard_forward_32t_20261003.json`](linux_x86_cpu_scorecard_forward_32t_20261003.json)

Install the exact optional comparison engines, hide CUDA, and run the two
forward profiles:

```bash
python3.12 -m venv --system-site-packages .venv
.venv/bin/python -m pip install -e '.[pennylane,qiskit]'

export CUDA_VISIBLE_DEVICES=""
export OMP_PROC_BIND=close OMP_PLACES=cores

taskset -c 32 .venv/bin/python -m \
  flagquantum.benchmarking.simulator_workload_corpus \
  --workloads hardware_efficient_statevector truncated_qft_statevector \
    random_clifford_statevector local_brickwork_statevector \
    dense_nonlocal_statevector swap_routing_statevector \
  --n-wires 18 22 \
  --engines flagquantum_native qiskit_aer pennylane_lightning_qubit \
  --threads 1 --warmup 2 --iterations 7 --calls-per-sample 1 \
  --json-output linux-x86-forward-1t.json

taskset -c 32-63 .venv/bin/python -m \
  flagquantum.benchmarking.simulator_workload_corpus \
  --workloads hardware_efficient_statevector truncated_qft_statevector \
    random_clifford_statevector local_brickwork_statevector \
    dense_nonlocal_statevector swap_routing_statevector \
  --n-wires 18 22 \
  --engines flagquantum_native qiskit_aer pennylane_lightning_qubit \
  --threads 32 --warmup 2 --iterations 7 --calls-per-sample 1 \
  --json-output linux-x86-forward-32t.json
```

## Boundaries, next action, and stopping condition

- The forward headline is one clean run on `jp-a800-172`. `jp-a800-171`
  supplied the accepted single-socket adjoint headline and `jp-a800-172`
  independently replicated that change. Results are not averaged across
  hosts.
- The forward evidence covers exact complex128 statevectors, 18/22 qubits,
  six deterministic circuits, and one process. It does not measure sampling,
  noisy simulation, process peak RSS, simultaneous-job throughput, SMT,
  multi-socket execution, complex64, or GPU behavior.
- Conversion is included for every forward engine. The table must not be used
  as an isolated native-kernel ranking or a universal framework ranking.
- This baseline stage stops because all requested cases have seven samples,
  every result passes correctness and stability, both single-thread and
  socket-local placement are recorded, absolute time and external ratios are
  visible, and the accepted adjoint evidence covers both hosts.
- The next PR measures multiple independent simulations per socket. It should
  compare one 32-thread job with several smaller pinned jobs at equal total
  core count, report tasks/second and tail latency, and make no runtime-policy
  change until the throughput result is reproducible.
