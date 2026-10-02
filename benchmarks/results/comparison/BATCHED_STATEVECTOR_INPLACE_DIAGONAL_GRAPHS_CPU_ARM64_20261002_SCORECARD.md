# In-place CPU diagonal graphs vs rollback and PennyLane Lightning

## Conclusion

For 18-qubit, batch-32, complex128 inference, reusing an executor-owned
statevector for fused controlled-phase and CZ graphs makes Truncated QFT
**1.018x faster** and Dense nonlocal **1.115x faster** than the exact functional
rollback. Dense nonlocal baseline-adjusted execution RSS falls from **392.2 MiB
to 311.0 MiB**, a **20.7% reduction**. QFT execution RSS changes from 324.8 MiB
to 321.0 MiB; that 1.1% difference is too small for a strong memory claim.

The same FlagQuantum path is **2.674x faster** than PennyLane Lightning native
batch on Truncated QFT and **2.307x faster** on Dense nonlocal. This is a focused
single-host comparison, not a universal simulator ranking.

## What is measured and why it matters

Both workloads return 32 exact statevectors with 262,144 complex128 amplitudes
each, so the logical result alone is 128 MiB. A fused diagonal graph previously
allocated another complete state tensor for every application. Once an earlier
kernel has produced an executor-owned tensor, the selected path now multiplies
the cached graph factors into that tensor in place. The initial-state cache is
never mutated, and any state requiring gradients remains on the functional path.

Warm time is the median of 11 complete calls after two warmups. Peak RSS is the
median of three separate cold Python processes per engine. Circuit construction
and one-time backend preprocessing are outside warm timing; state retrieval and
conversion to the common Torch result are included. All raw timing and RSS
observations are retained in the adjacent JSON artifact.

## Measured results

`Rollback speedup` is rollback time divided by selected time. `FQ vs Lightning`
is Lightning time divided by selected FlagQuantum time. Values above one mean
the selected FlagQuantum path is faster.

| Workload | Selected FQ | Functional rollback | Rollback speedup | PennyLane Lightning native batch | FQ vs Lightning | Selected execution RSS | Rollback execution RSS | RSS reduction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Truncated QFT | **759.850 ms** | 773.674 ms | **1.018x** | 2032.140 ms | **2.674x** | 321.0 MiB | 324.8 MiB | 1.1%; no strong claim |
| Dense nonlocal | **293.167 ms** | 326.862 ms | **1.115x** | 676.256 ms | **2.307x** | **311.0 MiB** | 392.2 MiB | **20.7%** |

Every timing distribution satisfies the 20% relative-MAD stability rule. The
selected path and exact rollback agree bit-for-bit in the recorded result;
PennyLane Lightning agrees within `1e-10`. The largest observed norm error is
`1.665e-14`.

Total process peak RSS is 501.0 vs 517.7 MiB for Truncated QFT and 504.2 vs
585.4 MiB for Dense nonlocal. Because interpreter and allocator baselines vary
between fresh processes, baseline-adjusted execution growth is the primary
temporary-memory measure.

## FlagQuantum user code

No public API change is required:

```python
import torch
import flagquantum as fq

angles = torch.linspace(-0.4, 0.4, 32, dtype=torch.float64)
circuit = fq.Circuit(18, bsz=32, dtype=torch.complex128)
for wire in range(18):
    circuit.h(wire)
for left in range(18):
    for right in range(left + 1, 18):
        circuit.cz(left, right)
for wire in range(18):
    circuit.ry(wire, angles + 0.01 * wire)

states = circuit.state()
assert states.shape == (32, 2**18)
```

The optimization is selected automatically for CPU inference after the
executor owns the state storage. Set `FQ_CPU_INPLACE_DIAGONAL_GRAPHS=0` for the
complete functional rollback used in this report.

## Reproduce

From a source checkout with PennyLane extras installed:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python -m flagquantum.benchmarking.batched_statevector_memory \
  --workloads truncated_qft_statevector dense_nonlocal_statevector \
  --n-wires 18 --batch-sizes 32 \
  --engines flagquantum_native_batch \
    flagquantum_native_diagonal_graph_rollback \
    pennylane_lightning_native_batch \
  --threads 1 --warmup 2 --iterations 11 --memory-probes 3 \
  --json-output batched-inplace-diagonal-graphs.json \
  --markdown-output BATCHED_INPLACE_DIAGONAL_GRAPHS.md
```

The complete machine-readable evidence is in the
[JSON artifact](batched_statevector_inplace_diagonal_graphs_cpu_arm64_20261002.json),
and the adjacent
[generated table](BATCHED_STATEVECTOR_INPLACE_DIAGONAL_GRAPHS_CPU_ARM64_20261002.md)
is rendered directly from that payload.

## Limits and next action

- Single Apple-arm64 host, one CPU thread, Python 3.12.14, PyTorch 2.13.0,
  PennyLane 0.45.1, and PennyLane Lightning 0.45.0 only.
- Exact complex128 statevector inference only; no gradients, shots, GPU,
  distributed, arbitrary-width, or scalability claim.
- In-place execution is used only for storage already owned by the executor.
  Cached inputs and autograd states retain functional semantics.
- The QFT execution-RSS difference is below a useful claim threshold even
  though total peak and median time improve.
- Local brickwork remains the largest measured statevector batch latency and
  memory gap; its mixed rotation/Clifford workspaces are the next CPU target.
