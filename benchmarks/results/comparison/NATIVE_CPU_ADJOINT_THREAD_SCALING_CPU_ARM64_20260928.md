# Native CPU adjoint thread scaling (Apple arm64 CPU)

This report measures FlagQuantum's exact `complex128` reversible statevector
adjoint after enabling the Torch OpenMP backend when compiling the package-local
C++ operators. Before this change the source used `at::parallel_for` and
`at::parallel_reduce`, but the extension was compiled without OpenMP, so those
templates selected their serial implementation. The public Python execution path
is unchanged.

Each value below is the median of seven retained value-and-full-gradient calls
after two warmups. Circuit construction and optimizer updates are excluded.
FlagQuantum and PennyLane Lightning use method-matched adjoint differentiation,
the same thread count, exact statevectors, and the same weighted Z/ZZ
Hamiltonians. All cases passed the `1e-9` value/gradient tolerance and the
maximum observed gradient error was `1.793e-12`.

## 22-qubit paired results

| Workload | CPU threads | FlagQuantum forward (ms) | FlagQuantum backward (ms) | FlagQuantum total (ms) | FlagQuantum speedup vs 1 thread | PennyLane Lightning total (ms) | PennyLane Lightning / FlagQuantum |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Hardware-efficient VQE | 1 | 502.984 | 667.626 | 1170.610 | 1.00x | 2205.962 | 1.88x |
| Hardware-efficient VQE | 2 | 344.884 | 318.164 | 680.957 | 1.72x | 1385.894 | 2.04x |
| Hardware-efficient VQE | 4 | 298.439 | 211.037 | 509.476 | 2.30x | 1442.755 | 2.83x |
| Hardware-efficient VQE | 8 | 284.274 | 170.933 | 454.850 | 2.57x | 1393.940 | 3.06x |
| QAOA path MaxCut | 1 | 409.375 | 489.652 | 907.105 | 1.00x | 904.832 | 1.00x |
| QAOA path MaxCut | 2 | 279.512 | 256.376 | 540.763 | 1.68x | 894.919 | 1.65x |
| QAOA path MaxCut | 4 | 216.645 | 155.755 | 376.790 | 2.41x | 939.186 | 2.49x |
| QAOA path MaxCut | 8 | 196.351 | 123.015 | 319.793 | 2.84x | 902.234 | 2.82x |

A PennyLane Lightning/FlagQuantum ratio above one means FlagQuantum was faster.
The result is local, non-release evidence for this host and workload rather than
a universal framework ranking. Four threads are a conservative default on this
10-logical-CPU host: they deliver 2.30x VQE and 2.41x QAOA speedups without
depending on the smaller incremental gain and lower parallel efficiency at eight
threads.

## Small-workload guardrail

The same seven-sample sweep checks that parallel dispatch does not erase the
benefit at 18 qubits.

| Workload | CPU threads | FlagQuantum total (ms) | Speedup vs 1 thread |
| --- | ---: | ---: | ---: |
| Hardware-efficient VQE | 1 | 50.978 | 1.00x |
| Hardware-efficient VQE | 2 | 30.586 | 1.67x |
| Hardware-efficient VQE | 4 | 21.889 | 2.33x |
| Hardware-efficient VQE | 8 | 30.787 | 1.66x |
| QAOA path MaxCut | 1 | 46.817 | 1.00x |
| QAOA path MaxCut | 2 | 24.425 | 1.92x |
| QAOA path MaxCut | 4 | 23.135 | 2.02x |
| QAOA path MaxCut | 8 | 21.563 | 2.17x |

## Kernel diagnosis

A direct 22-qubit rotation-segment operator probe used four retained calls. With
the old build, 1/2/4/8 threads took 322.382/325.476/329.033/322.356 ms and
process CPU time stayed at approximately one core. After rebuilding with the
Torch OpenMP backend, the same probe took 364.185/190.364/108.240/86.689 ms;
four threads were 3.36x faster than the new one-thread measurement and consumed
approximately 3.8 cores. This isolates the fixed defect from Python planning and
the rest of the statevector execution path.

## FlagQuantum example

```python
import torch
import flagquantum as fq
from flagquantum import algorithms as fqa

torch.set_num_threads(4)
theta = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
circuit = fq.Circuit(3, dtype=torch.complex128).ry(0, theta).cx(0, 1)
hamiltonian = fqa.Hamiltonian((
    fqa.pauli_term(0.7, "ZZ", (0, 1)),
    fqa.pauli_term(0.2, "Z", (2,)),
))
energy = hamiltonian.expectation(circuit, differentiation="adjoint")
energy.backward()
print(energy.item(), theta.grad)
```

## Reproduction

Build the package-local extension, then run each thread count in a fresh process.
Fresh processes keep the OpenMP environment and framework initialization
identical for each row.

```bash
python setup.py build_ext --inplace --force

for threads in 1 2 4 8; do
  OMP_NUM_THREADS=$threads MKL_NUM_THREADS=$threads \
  flagquantum-benchmark run differentiable_simulator_corpus \
    --engines flagquantum_adjoint pennylane_lightning_adjoint \
    --workloads hardware_efficient_vqe qaoa_path_maxcut \
    --n-wires 22 --layers 1 --threads $threads \
    --warmup 2 --iterations 7 --calls-per-sample 1 \
    --skip-memory-probe \
    --json-output adjoint-thread-scaling-$threads.json
done
```

The 18-qubit guardrail uses the same command with
`--engines flagquantum_adjoint --n-wires 18`.
