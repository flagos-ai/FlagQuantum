# CPU statevector batch-window scorecard

## Conclusion

FlagQuantum now executes wide CPU parameter batches as complete-program windows
whose logical statevectors fit a 64 MiB budget. On the measured Apple arm64
host, this reduced total time for every 18-qubit, batch-32 workload relative to
the same native batch with windowing disabled: `1.05x` to `1.27x`, with a
`1.12x` geometric-mean speedup across the five workloads.

This removes part of the wide-batch regression; it does not make native batch
universally optimal. The budgeted path remained `6.6%` to `18.8%` slower than
32 independent FlagQuantum scalar executions in this run. PennyLane Lightning
was fastest on hardware-efficient, random-Clifford, and local-brickwork;
Cirq was narrowly fastest on truncated QFT; FlagQuantum was fastest on dense
nonlocal. The result supports merging the bounded-working-set change while
leaving further batch-kernel work on the CPU roadmap.

## What was measured

Each row is one complete user task returning 32 exact complex128 statevectors
for 32 independent parameter bindings of the same 18-qubit circuit structure.
Circuit construction is excluded. Backend preparation, execution, and result
retrieval are included. PyTorch, OpenMP, MKL, and OpenBLAS use one CPU thread.

The monolithic row is the exact FlagQuantum rollback path
(`FQ_CPU_STATEVECTOR_BATCH_CHUNKING=0`) measured in the same process. External
bridges currently accept one circuit item, so their task time is 32 repeated
public bridge calls. This is a fair comparison of the FlagQuantum user-facing
task available today, not a claim about every framework's best private or
native batching API.

| Workload | FQ budgeted batch (ms) | FQ monolithic batch (ms) | Window speedup | FQ serial (ms) | Qiskit Aer bridge (ms) | Cirq bridge (ms) | PennyLane Lightning bridge (ms) | Fastest framework path |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Hardware-efficient | 705.031 | 776.374 | 1.10x | 593.294 | 1590.864 | 1834.387 | 641.079 | PennyLane Lightning |
| Truncated QFT | 1290.929 | 1356.132 | 1.05x | 1100.812 | 3101.784 | 1250.088 | 1813.558 | Cirq |
| Random Clifford | 1338.473 | 1516.169 | 1.13x | 1255.735 | 1930.114 | 1166.824 | 505.141 | PennyLane Lightning |
| Local brickwork | 1281.584 | 1344.053 | 1.05x | 1091.893 | 1718.701 | 2235.310 | 984.076 | PennyLane Lightning |
| Dense nonlocal | 349.150 | 442.847 | 1.27x | 291.855 | 2716.033 | 1411.837 | 458.637 | FlagQuantum budgeted batch |

All 30 engine/workload comparisons passed the `1e-10` absolute-error contract.
All measurements passed the `0.20` relative median absolute deviation limit.
The JSON contains all seven timed samples, package versions, exact errors, and
environment metadata.

## User code

Windowing is automatic; the public circuit and statevector API does not change.
For complex128 at 18 qubits, one row is 4 MiB, so the 64 MiB policy executes
this batch in two 16-row windows and concatenates the differentiable results.

```python
import torch
import flagquantum as fq

batch = 32
qubits = 18
angles = torch.linspace(
    -0.4, 0.4, batch, dtype=torch.float64, requires_grad=True
)

circuit = fq.Circuit(qubits, bsz=batch, dtype=torch.complex128)
for wire in range(qubits):
    circuit.ry(wire, angles + 0.01 * wire)
    circuit.cx(wire, (wire + 1) % qubits)

states = circuit.state()  # shape: [32, 2**18]
loss = states.real.square().sum()
loss.backward()
```

## Reproduce

```bash
pip install -e '.[qiskit,cirq,pennylane]'
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
flagquantum-benchmark run batched_statevector_corpus \
  --workloads hardware_efficient_statevector truncated_qft_statevector \
    random_clifford_statevector local_brickwork_statevector \
    dense_nonlocal_statevector \
  --n-wires 18 --batch-sizes 32 \
  --engines flagquantum_native_batch \
    flagquantum_native_monolithic_batch flagquantum_native_serial \
    qiskit_aer_bridge cirq_simulator_bridge pennylane_lightning_bridge \
  --threads 1 --warmup 2 --iterations 7 \
  --json-output batched-statevector-chunking.json \
  --markdown-output batched-statevector-chunking.md
```

Raw evidence:
[`batched_statevector_chunking_cpu_arm64_20261001.json`](batched_statevector_chunking_cpu_arm64_20261001.json).
Generated complete table:
[`BATCHED_STATEVECTOR_CHUNKING_CPU_ARM64_20261001.md`](BATCHED_STATEVECTOR_CHUNKING_CPU_ARM64_20261001.md).

## Limits

- This is one Apple arm64 CPU and one thread; it is comparison evidence, not a
  universal framework ranking or release/scalability evidence.
- The 64 MiB value budgets logical state rows, not process peak RSS or every
  kernel temporary.
- External bridge times include repeated public conversions and do not measure
  framework-private native batching.
- Wide native batch still trails FlagQuantum scalar serial here. The next CPU
  phase should reduce per-window kernel and concatenation overhead rather than
  increasing the budget until it becomes monolithic again.
