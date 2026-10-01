# CPU batched statevector memory scorecard

## Conclusion

The 64 MiB execution windows introduced in #297 reduce real process memory as
well as logical working-set size. On this Apple arm64 host, the budgeted
FlagQuantum path used `17.7%` to `27.7%` less peak RSS than the same native batch
with windowing disabled, while also completing the task `1.13x` to `1.33x`
faster (`1.18x` geometric mean across the five workloads).

This closes the missing real-memory validation for #297; it does not establish
that native batching is memory-optimal. Budgeted FlagQuantum still used
`1.70x` to `1.90x` the peak RSS of 32 independent FlagQuantum scalar calls, and
was `1.08x` to `1.23x` slower than that serial path. The next batch-memory phase
should reduce per-window temporary tensors rather than enlarge the window.

## What was measured

Every task returns 32 exact complex128 statevectors for independent parameter
bindings of one 18-qubit circuit structure. Warm timings include backend
execution, per-call conversion, and result retrieval; they exclude circuit
construction and one-time engine initialization. They use 2 warmups and 11
measured samples. PyTorch, OpenMP, MKL, and OpenBLAS use one CPU thread.

Peak RSS is measured separately from timing: each engine/workload starts in a
fresh interpreter and performs one cold, complete task. The high-water resident
set comes from `resource.getrusage(RUSAGE_SELF).ru_maxrss`; framework imports,
the interpreter baseline, circuit construction, execution, and retained output
are all included. This makes peaks comparable between rows without one engine
contaminating another's process high-water mark.

| Workload | FQ budgeted time (ms) | FQ monolithic time (ms) | Speedup | FQ budgeted peak (MiB) | FQ monolithic peak (MiB) | Peak reduction | FQ serial time / peak | Best external time / peak |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| Hardware-efficient | 813.310 | 930.222 | 1.14x | 1042.5 | 1363.3 | 23.5% | 668.579 ms / 607.4 MiB | PennyLane Lightning: 727.017 ms / 637.4 MiB |
| Truncated QFT | 863.867 | 986.409 | 1.14x | 1166.1 | 1416.3 | 17.7% | 700.964 ms / 618.4 MiB | Cirq: 915.210 ms / 627.4 MiB |
| Random Clifford | 1074.345 | 1213.435 | 1.13x | 1167.6 | 1615.5 | 27.7% | 969.539 ms / 616.1 MiB | PennyLane Lightning: 404.360 ms / 671.2 MiB |
| Local brickwork | 934.087 | 1107.615 | 1.19x | 1042.8 | 1363.4 | 23.5% | 857.850 ms / 612.2 MiB | PennyLane Lightning: 863.757 ms / 476.1 MiB |
| Dense nonlocal | 439.069 | 585.853 | 1.33x | 1042.0 | 1362.1 | 23.5% | 407.270 ms / 612.7 MiB | PennyLane Lightning: 530.593 ms / 671.7 MiB |

Qiskit Aer had the lowest consistent process peaks (`496.0–497.4 MiB`) but was
slower than budgeted FlagQuantum on all five tasks (`1.50x–7.02x` as long).
Cirq used `571.4–630.4 MiB`; PennyLane Lightning used `476.1–671.7 MiB`.
These are public FlagQuantum bridge tasks, not framework-private native batching
benchmarks.

All 30 engine/workload comparisons passed the `1e-10` absolute-error contract.
All 30 timing distributions passed the `0.20` relative median absolute
deviation limit. The raw JSON retains all 11 samples, exact errors, output
normalization errors, versions, and process-memory fields.

## User code

Memory windowing remains automatic; users keep the regular FlagQuantum API:

```python
import torch
import flagquantum as fq

batch = 32
qubits = 18
angles = torch.linspace(-0.4, 0.4, batch, dtype=torch.float64)

circuit = fq.Circuit(qubits, bsz=batch, dtype=torch.complex128)
for wire in range(qubits):
    circuit.ry(wire, angles + 0.01 * wire)
    circuit.cx(wire, (wire + 1) % qubits)

states = circuit.state()  # [32, 2**18], executed as bounded CPU windows
```

## Reproduce

```bash
pip install -e '.[qiskit,cirq,pennylane]'
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
flagquantum-benchmark run batched_statevector_memory \
  --workloads hardware_efficient_statevector truncated_qft_statevector \
    random_clifford_statevector local_brickwork_statevector \
    dense_nonlocal_statevector \
  --n-wires 18 --batch-sizes 32 \
  --engines flagquantum_native_batch \
    flagquantum_native_monolithic_batch flagquantum_native_serial \
    qiskit_aer_bridge cirq_simulator_bridge pennylane_lightning_bridge \
  --threads 1 --warmup 2 --iterations 11 \
  --json-output batched-statevector-memory.json \
  --markdown-output batched-statevector-memory.md
```

Raw evidence:
[`batched_statevector_memory_cpu_arm64_20261001.json`](batched_statevector_memory_cpu_arm64_20261001.json).
Generated complete table:
[`BATCHED_STATEVECTOR_MEMORY_CPU_ARM64_20261001.md`](BATCHED_STATEVECTOR_MEMORY_CPU_ARM64_20261001.md).

## Limits

- This is one Apple arm64 host, one CPU thread, one dtype, one width, and one
  batch size. It is local comparison evidence, not a universal ranking or
  release/scalability evidence.
- Peak RSS includes each framework's process/import baseline and circuit
  construction. The execution-growth column in the complete table subtracts
  the pre-execution high-water mark, but it is still a high-water delta rather
  than allocator-level attribution.
- Timing and RSS are deliberately separate runs: timing is warm and rotated in
  one process; memory is a cold task in a fresh process.
- External bridges execute 32 public single-item calls because they do not yet
  expose native batch execution through the FlagQuantum bridge.
- The budgeted path still consumes substantially more memory than FlagQuantum
  serial execution, so the CPU batch-memory roadmap is not complete.
