# Linux x86-64 GCC 11 native CPU adjoint build and performance

## Decision

**The native CPU adjoint extension now builds correctly with GCC 11 and
OpenMP on Ubuntu 22.04.** The unmodified `main` snapshot at
`02ea7db5818762237be75f2c097d3909a5e07559` fails while compiling
`rotation_adjoint.cpp`: GCC 11 cannot resolve the reduction variables in six
OpenMP SIMD directives nested inside the ATen dispatch and parallel lambdas.

The compatibility path leaves those six loops scalar on GCC 11 and older. It
does not disable the native extension, ATen parallelism, or OpenMP elsewhere.
GCC 12+ and Clang retain the explicit SIMD directives.

The fix was built independently on two identical Linux x86-64 nodes. Both
reported `native_cpu_adjoint_available() == True` and
`native_cpu_parallel_build_available() == True`, and both passed all 143
targeted native adjoint tests. The measured node also retained a clear
end-to-end lead over PennyLane Lightning adjoint in all four measured cases.

## Environment

| Field | Value |
| --- | --- |
| CPU | 2 x Intel Xeon Platinum 8358, 32 physical cores per socket, SMT enabled |
| NUMA | 2 nodes; timing pinned to logical CPU 0 on NUMA node 0 |
| Memory | 1 TiB |
| OS | Ubuntu 22.04, Linux 5.15, x86-64 |
| Compiler | GCC/G++ 11.4.0, OpenMP 4.5 |
| Python | 3.12.13 |
| PyTorch | 2.13.0+cu130, CPU device, one thread |
| FlagQuantum source | `02ea7db5818762237be75f2c097d3909a5e07559` plus this compatibility patch |
| Source archive SHA-256 | `7c98cab6917ef3bef688ce9a3a15a60c109007af520986304a29c2409622a716` |

CUDA devices were hidden from the benchmark process. The primary timing metric
is expectation value plus the full parameter gradient. Construction and
parameter initialization are outside the timed region. Each cell is the median
of seven retained calls after two warmups; engines rotate within one process.

## Measured comparison

A PennyLane Lightning/FlagQuantum ratio above one means FlagQuantum is faster.
Only the complete value-and-gradient time is compared across framework timing
boundaries.

| Workload | Qubits | FlagQuantum value + gradient | PennyLane Lightning adjoint | Lightning / FlagQuantum | FlagQuantum relative MAD | Lightning relative MAD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Hardware-efficient VQE | 18 | **42.525 ms** | 190.456 ms | **4.479x** | 2.24% | 0.98% |
| QAOA path MaxCut | 18 | **49.222 ms** | 128.460 ms | **2.610x** | 0.62% | 0.32% |
| Hardware-efficient VQE | 22 | **760.967 ms** | 4,425.672 ms | **5.816x** | 0.58% | 0.19% |
| QAOA path MaxCut | 22 | **942.468 ms** | 2,800.858 ms | **2.972x** | 0.55% | 0.09% |

All four cases pass the 20% relative-MAD stability rule. Values and full
gradients agree within the corpus tolerance of `1e-9`; the largest observed
PennyLane gradient error is `3.082e-13` and the largest value error is
`1.036e-11`.

Raw measurements retain every timing sample, correctness error, workload hash,
runtime version, and methodology field:

- [`linux_x86_gcc11_adjoint_18q_1t_20261003.json`](linux_x86_gcc11_adjoint_18q_1t_20261003.json)
- [`linux_x86_gcc11_adjoint_22q_1t_20261003.json`](linux_x86_gcc11_adjoint_22q_1t_20261003.json)

## Reproduce

Use Ubuntu 22.04 with its default GCC 11 compiler and install the project with
the PennyLane extra. The package build is part of the reproduction: unmodified
`main` fails before the benchmark command, while this patch builds the native
extension.

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install \
  "torch==2.13.*" --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -e '.[dev,pennylane]'

export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export OMP_PROC_BIND=close OMP_PLACES=cores

taskset -c 0 .venv/bin/python -m \
  flagquantum.benchmarking.differentiable_simulator_corpus \
  --workloads hardware_efficient_vqe qaoa_path_maxcut \
  --n-wires 18 22 --layers 1 \
  --engines flagquantum_adjoint pennylane_lightning_adjoint \
  --threads 1 --warmup 2 --iterations 7 --calls-per-sample 1 \
  --skip-memory-probe --json-output linux-x86-adjoint.json

taskset -c 0-31 .venv/bin/python -m pytest \
  tests/unit/test_native_cpu_adjoint.py -q
```

## Boundaries and next step

- The performance table is one pinned socket-local CPU run. The second node is
  independent build-and-correctness evidence, not an additional timing sample.
- This result certifies GCC 11 buildability and single-thread adjoint behavior;
  it does not establish multi-socket scaling, NUMA policy, forward performance,
  batch throughput, peak memory, GPU performance, or a universal framework
  ranking.
- Disabling explicit SIMD on GCC 11 is a compatibility fallback. A future
  Linux optimization may recover portable vectorization, but only if a
  matched end-to-end A/B measurement beats this accepted baseline without
  weakening correctness or compiler coverage.

