# Transverse-Field Ising VQE: JAX JIT Benefits Across Circuit Depths

## End-to-End Training Trajectories: Adam, 500 Steps

Beyond one-step benchmarks, `benchmarks/jax_jit_vqe_training.py` runs PyTorch Native
and JAX JIT with identical initialization, Hamiltonians, ansatzes, Adam settings,
and 500 actual optimizer steps. JAX wall time includes kernel construction and
initial compilation; each step includes forward, backward, optimizer update, and
device synchronization. SciPy sparse `eigsh` supplies exact ground-state energies.
The convergence threshold is energy error `1e-3`.

| Qubits | Depth | Converges within 500 steps | First threshold step | JAX cumulative break-even step |
| ---: | ---: | :---: | ---: | ---: |
| 4 | 2 | Yes | 175 | 136 |
| 4 | 4 | Yes | 92 | 157 |
| 4 | 8 | Yes | 72 | 141 |
| 8 | 2 | No | — | 145 |
| 8 | 4 | No | — | 131 |
| 8 | 8 | Yes | 110 | 122 |
| 12 | 2 | No | — | 147 |
| 12 | 4 | No | — | 108 |
| 12 | 8 | No | — | 120 |

JAX eventually breaks even in cumulative time for every configuration, but
performance break-even is not algorithm convergence. Shallow 8-qubit circuits and
all 12-qubit configurations fail to reach `1e-3` under the tested ansatz,
initialization, Adam learning rate, and 500-step budget. Both backends reach the
threshold at the same steps, indicating that JAX mainly changes speed while
preserving per-step optimization behavior.

The L-BFGS path is implemented and records closure evaluations. A800 execution
was not authorized in this round, so L-BFGS is not labeled measured evidence.

## Objective

Use real VQE energy+gradient workloads to determine which qubit counts and circuit
depths benefit from JAX JIT, the initial compilation cost, and break-even reuse counts.

## Frozen Workload

- Hamiltonian: `H = -0.7 Σ ZiZi - 0.25 Σ Xi`, with open boundaries.
- Ansatz: RX/RY/RZ, nearest-neighbor CX, and first-to-last RXX in every layer.
- Simulation: exact statevector, complex64.
- Training computation: Hamiltonian energy plus backward gradients for all parameters.
- Grid: qubits `{4, 8, 12, 16, 20}` × depth `{2, 4, 8}`.
- Backends: FlagQuantum PyTorch Native and JAX JIT quantum kernels through PyTorch.
- Cold starts: fresh Python processes per point, persistent compilation cache disabled.
- Semantics: `single_device_fast_path`, not distributed scalability evidence.

## A800 Measurements

Environment: one NVIDIA A800-SXM4-80GB, PyTorch 2.13.0+cu130, JAX 0.10.2.

| Qubits | Depth | JAX cold start | PyTorch steady | JAX steady | Speedup | Break-even steps |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 2 / 4 / 8 | 4.5 / 8.3 / 13.7 s | 26.7 / 46.7 / 82.6 ms | 1.0 / 1.6 / 2.8 ms | 26.9 / 29.2 / 29.4× | 171 / 182 / 171 |
| 8 | 2 / 4 / 8 | 8.2 / 13.4 / 22.8 s | 45.0 / 87.0 / 164.3 ms | 1.9 / 3.2 / 6.3 ms | 24.2 / 27.1 / 26.3× | 190 / 159 / 145 |
| 12 | 2 / 4 / 8 | 12.2 / 15.9 / 31.9 s | 66.9 / 127.0 / 238.9 ms | 3.0 / 5.5 / 10.7 ms | 22.5 / 23.1 / 22.4× | 190 / 131 / 140 |
| 16 | 2 / 4 / 8 | 14.1 / 22.9 / 44.9 s | 99.2 / 170.3 / 325.0 ms | 9.2 / 16.9 / 27.0 ms | 10.8 / 10.1 / 12.1× | 156 / 149 / 151 |
| 20 | 2 / 4 / 8 | 17.7 / 28.4 / 50.6 s | 126.6 / 216.3 / 417.5 ms | 93.9 / 163.6 / 318.0 ms | 1.35 / 1.32 / 1.31× | 534 / 533 / 505 |

All 15 points pass the `1e-4` loss/gradient correctness threshold.

## Findings

1. At 4–12 qubits, JAX JIT provides clear steady-state gains of roughly 22–29×.
2. At 16 qubits, gains remain about 10–12×, but initial compilation reaches 44.9 seconds.
3. At 20 qubits, gains narrow to roughly 1.3×, cold start reaches 50.6 seconds, and
   break-even requires about 505–534 same-shape energy+gradient calls.
4. Planners should consider depth, expected training steps, shape stability, and
   compilation-cache hits as well as qubit count.

## Figure Sequence

1. `01_vqe_jax_benefit_map.svg`: steady-state benefit regions.
2. `02_vqe_compile_cost_map.svg`: initial compilation cost across size/depth.
3. `03_vqe_break_even_map.svg`: cumulative-call break-even boundaries.
4. `04_vqe_backend_decision_map.svg`: backend choices combining benefit and compilation cost.

These are development measurements with one cold start, suitable for case studies
and planner hypotheses. Formal release claims need at least three independent
cold starts, confidence intervals, and replication on a second machine.
