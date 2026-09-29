# Native CPU dual-state CX adjoint gather

## Conclusion

This change accelerates the CX-permutation part of FlagQuantum's native CPU
adjoint sweep. It does not make the complete simulator scale linearly with the
thread count: rotation arithmetic, observable construction, allocator work,
memory bandwidth, and serial orchestration remain in the full VQE/QAOA path.

The optimized kernel reads one cached CX permutation table and gathers the ket
and adjoint in the same parallel C++ loop. The rollback path launches two
`torch.index_select` operations and then copies both results back into the
reversible buffers. Exact complex128 outputs matched bit for bit.

## What is measured and why it matters

The workload is one 22-qubit, batch-one, complex128 ket plus one equally sized
adjoint. Each state is 64 MiB. A chain of 21 CX gates is represented by one
cached 16 MiB int32 inverse-permutation table. This is the same shape used by
the one-layer 22-qubit hardware-efficient VQE and path-MaxCut QAOA comparison
workloads.

The baseline deliberately times only the two `index_select` calls; it omits the
two subsequent copy-backs performed by the old reverse sweep. The reported
speedup is therefore conservative for the replaced region.

| CPU threads | Two PyTorch gathers (ms) | Fused C++ dual gather (ms) | Kernel speedup | Baseline rMAD | Native rMAD |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 18.560 | 4.820 | 3.85x | 2.44% | 5.32% |
| 4 | 18.944 | 3.595 | 5.27x | 0.60% | 5.35% |
| 8 | 18.595 | 3.446 | 5.40x | 1.78% | 3.10% |

The important result is not an 8x simulator claim. It is that the previously
almost thread-insensitive permutation region now uses the native parallel
backend and falls from 18.6 ms at one thread to 3.45 ms at eight threads. The
complete 22-qubit corpus remains subject to host load and to the other serial
and bandwidth-bound phases, so this artifact is classified as a local,
non-release microbenchmark rather than a universal framework ranking.

## User-level example

No public API changes. Existing adjoint code automatically uses the kernel for
eligible CPU CX sequences:

```python
import torch
import flagquantum as fq
from flagquantum import algorithms as fqa

theta = torch.tensor(0.31, dtype=torch.float64, requires_grad=True)
circuit = fq.Circuit(22, device="cpu", dtype=torch.complex128)
for wire in range(22):
    circuit.ry(wire, theta)
for wire in range(21):
    circuit.cx(wire, wire + 1)

hamiltonian = fqa.Hamiltonian((fqa.pauli_term(1.0, "Z", (0,)),))
energy = hamiltonian.expectation(circuit, differentiation="adjoint")
energy.backward()
print(energy.item(), theta.grad)
```

Set `FQ_NATIVE_CPU_CX_ADJOINT_GATHER=0` to restore the two-gather PyTorch path.

## Reproduction

```bash
python setup.py build_ext --inplace --force

PYTHONPATH=. OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
python benchmarks/native_cpu_cx_adjoint_gather.py \
  --n-wires 22 --threads 1 4 8 \
  --warmup 3 --iterations 12 --calls-per-sample 5 \
  --json-output native-cx-adjoint-gather.json
```

The checked raw artifact is
[`native_cpu_cx_adjoint_gather_cpu_arm64_20260929.json`](native_cpu_cx_adjoint_gather_cpu_arm64_20260929.json).
It records Python 3.12.14, PyTorch 2.13.0, macOS arm64, every timed sample,
correctness status, state size, thread count, and methodology.

## Known limits

- CPU, contiguous complex64/complex128 states, and int32/int64 permutation
  tables only.
- The fast path is private to reversible adjoint execution; forward execution
  and public circuit semantics are unchanged.
- This result does not establish production performance, cross-host
  reproducibility, distributed scaling, or a release gate.
