# Linux x86-64 CPU simulator roadmap closeout

## Decision

**Complete the defined Linux x86-64 CPU tranche.** The post-merge workflow on
`main` reproduced the optimized 22-qubit Random Clifford path on the fixed
Xeon runner, stayed within the internal regression budget, and remained faster
than both maintained same-semantics external engines. The earlier baseline,
socket-local throughput, NUMA traffic, measured-hotspot kernel, and recurring
regression-gate stages all have reproducible evidence and explicit boundaries.

This decision closes the declared tranche; it is not a claim that FlagQuantum
wins every circuit, qubit count, thread count, or CPU architecture. The older
full-corpus scorecard deliberately retains its single-circuit latency losses.
Those are future workload-specific optimization candidates, not erased by this
bounded closeout.

## Post-merge result on `main`

GitHub Actions run
[`37292570952`](https://github.com/flagos-ai/FlagQuantum/actions/runs/37292570952)
measured source revision `52314f6932d040bf3dbe78383b374fa1a144694a` on one
32-core socket of an Intel Xeon Platinum 8358. The workload is the maintained
22-qubit, complex128 Random Clifford circuit with 132 gates and logical depth 8.
The timed region includes conversion, backend preparation, execution, and full
statevector retrieval. Two warmups precede seven retained calls.

| Engine | Version | Median | RMAD | FlagQuantum advantage |
| --- | --- | ---: | ---: | ---: |
| **FlagQuantum native** | 0.2.0 | **47.492 ms** | 6.37% | 1.000x |
| Qiskit Aer | Qiskit 2.5.1 / Aer 0.17.2 | 141.778 ms | 8.52% | **2.985x** |
| PennyLane Lightning | PennyLane 0.45.1 / Lightning 0.45.0 | 441.017 ms | 0.73% | **9.286x** |

The FlagQuantum median is `1.027x` the checked 46.257 ms internal baseline,
inside the allowed `1.20x` ceiling. Every engine passed the 20% RMAD rule and
exact-statevector correctness at absolute tolerance `1e-10`; the largest
external error was `9.11e-18`. Both external performance floors passed.

Raw samples and the exact environment are preserved in
[`linux_x86_cpu_roadmap_closeout_main_20261005.json`](linux_x86_cpu_roadmap_closeout_main_20261005.json).

## Completed stages

| Stage | Authoritative evidence | Result |
| --- | --- | --- |
| Formal Linux x86 baseline | [`LINUX_X86_CPU_SCORECARD_20261003.md`](LINUX_X86_CPU_SCORECARD_20261003.md) | Six exact-statevector structures at 18/22 qubits, 1/32 threads, plus matched adjoint evidence |
| Socket-local independent jobs | [`SOCKET_LOCAL_MULTITASK_THROUGHPUT_LINUX_X86_20261003.md`](SOCKET_LOCAL_MULTITASK_THROUGHPUT_LINUX_X86_20261003.md) | Best FlagQuantum throughput is **1.228x–8.119x** the best external engine across all six maintained workloads |
| NUMA memory traffic | [`NUMA_MEMORY_TRAFFIC_LINUX_X86_20261004.md`](NUMA_MEMORY_TRAFFIC_LINUX_X86_20261004.md) | Keep first-touch; forced bind/interleave did not improve every workload |
| Measured maximum hotspot | [`NATIVE_CPU_PRODUCT_MIXED_CLIFFORD_LINUX_X86_20261004.md`](NATIVE_CPU_PRODUCT_MIXED_CLIFFORD_LINUX_X86_20261004.md) | Portable native mixed-Clifford route is **3.24x** faster than exact rollback and faster than both external engines |
| Recurring performance gate | [`LINUX_X86_CPU_REGRESSION_GATE_20261004.md`](LINUX_X86_CPU_REGRESSION_GATE_20261004.md) | Weekly/manual fixed-runner gate passes on post-merge `main` with isolated pinned dependencies |

## Public path

No benchmark-only backend is required. The maintained gate exercises the normal
CPU statevector API:

```python
import random

import flagquantum as fq
import torch

n_qubits = 22
generator = random.Random(7319 + 10_007 * n_qubits)
circuit = fq.Circuit(n_qubits, dtype=torch.complex128)
for _ in range(4):
    for qubit in range(n_qubits):
        getattr(circuit, generator.choice(("h", "s", "x")))(qubit)
    qubits = list(range(n_qubits))
    generator.shuffle(qubits)
    for index in range(0, n_qubits - 1, 2):
        getattr(circuit, generator.choice(("cx", "cz")))(
            qubits[index], qubits[index + 1]
        )

state = circuit.state()
assert state.shape == (2**n_qubits,)
```

## Reproduce

The repository workflow is the authoritative reproduction because it provisions
the exact external versions in a run-specific virtual environment, pins the
process to CPUs `0-31`, uploads all raw gate artifacts, and removes the temporary
environment even after failure:

```bash
gh workflow run linux-x86-cpu-performance.yml \
  --repo flagos-ai/FlagQuantum --ref main
```

The equivalent benchmark command inside that environment is:

```bash
OMP_NUM_THREADS=32 MKL_NUM_THREADS=32 OPENBLAS_NUM_THREADS=32 \
taskset -c 0-31 flagquantum-benchmark run simulator_workload_corpus \
  --workloads random_clifford_statevector --n-wires 22 \
  --engines flagquantum_native qiskit_aer pennylane_lightning_qubit \
  --threads 32 --warmup 2 --iterations 7 --calls-per-sample 1 \
  --json-output current.json
```

## Boundaries and restart condition

- This closeout is local, exact-statevector, complex128, single-process evidence
  on one Linux x86-64 CPU model. It is not distributed, noisy, approximate,
  GPU, capacity, or universal framework-ranking evidence.
- The maintained recurring gate covers one optimized hotspot. The broader
  scorecard remains diagnostic evidence and contains workloads where Aer or
  Lightning had lower single-circuit latency at the recorded older revision.
- Socket-local throughput and single-circuit latency are different objectives;
  neither result may be substituted for the other.
- A new CPU optimization tranche should start only from a reproducible user
  workload with a same-semantics reference, correctness tolerance, fixed timing
  boundary, rollback, success threshold, and stop condition.
- Reopen this tranche if the recurring gate regresses beyond `1.20x` of the
  internal baseline, FlagQuantum loses either maintained external floor, the
  case matrix or correctness checks shrink, or a supported Linux x86 profile
  becomes incomparable.

The tranche stops here because every declared stage is present, the post-merge
workflow passed on `main`, all seven-sample timing groups are stable, the exact
statevectors agree, both external floors pass, and no concrete defect was found.
