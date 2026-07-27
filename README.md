<div align="center">
  <img src="assets/logo.png" alt="FlagQuantum Logo" width="380">
</div>

# FlagQuantum

FlagQuantum is a quantum AI framework centered on one public API:

```python
import flagquantum as fq
```

The current implementation provides a FlagQuantum-native circuit IR, PyTorch
training interfaces, local statevector/MPS/tensor-network runtimes, optional
JAX quantum kernels, deployment packaging, and auditable distributed planning.
Distributed scalability claims are intentionally fail-closed: replicated
per-rank execution is not described as capacity scaling.

## Documentation Entry Points

## Source architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for the runtime and benchmark layout.
User-facing imports belong in `flagquantum.api`; reproducible benchmark
drivers belong in `flagquantum.benchmarking`; exploratory plots stay in
`benchmarks.research`.

- [FlagQuantum vision](docs/FLAGQUANTUM_VISION.md)
- [Ecosystem development strategy](docs/ECOSYSTEM_DEVELOPMENT.md)
- [API reference](docs/API.md)
- [Capability maturity](docs/CAPABILITY_MATURITY.md)
- [Known limitations](docs/KNOWN_LIMITATIONS.md)
- [Distributed quantum AI principles](docs/DISTRIBUTED_QUANTUM_AI_PRINCIPLES.md)
- [Distributed scalability principles](docs/DISTRIBUTED_SCALABILITY_PRINCIPLES.md)
- [Exascale and Gordon Bell strategy](docs/EXASCALE_GORDON_BELL_STRATEGY.md)
- [Feature parity matrix](docs/FEATURE_PARITY_MATRIX.md)
- [Runnable examples](examples/README.md)
- [Tutorial notebooks](examples/tutorials/README.md)
- [Release notes](docs/RELEASE_NOTES.md)

[![PyTorch](https://img.shields.io/badge/PyTorch-2.5%2B-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10--3.12-blue.svg)](https://python.org/)

## Current Capability Surface

- **Unified circuit API and IR**: build circuits with `fq.Circuit`, export
  FlagQuantum IR, draw circuits, compile to topology constraints, and run
  statevector/MPS/TN/density-matrix modes.
- **PyTorch-first training**: circuit execution and trainable examples are
  exposed through PyTorch tensors, optimizers, and autograd-compatible paths.
- **Optional JAX kernels**: compatibility modules expose JAX
  value-and-gradient kernels through a PyTorch-facing interface when JAX is
  installed; they are not part of the stable root API.
- **Runtime planning**: `fq.plan_runtime_selection(...)` and
  `Circuit.runtime_plan(...)` report candidate modes, memory plans,
  communication plans, gradient support, deployment readiness, blockers, and
  distribution semantics.
- **Distributed guardrails**: audit-policy compatibility modules check metadata
  honesty and fail closed before evidence can be considered for release.
- **Deployment packaging**: circuits can be compiled into portable deployment
  packages and routed through provider abstractions or exported formats.

## Distributed Claim Boundary

FlagQuantum uses these semantics in runtime summaries and benchmark JSON:

- `single_device_fast_path`: local CPU or one-GPU execution. This can be a
  performance result, but not a distributed scalability result.
- `rank_local_replicated_kernel`, `data_parallel_replicated`, or
  `replicated_per_rank`: useful for throughput, smoke tests, or rank-local
  acceleration, but not one-workload capacity scaling.
- `manual_sliced_tensor_contraction`: sliced TN work that must still report
  blockers before it is promoted as full distributed training scalability.
- `sharded_across_ranks`: one logical workload is partitioned across ranks.
  This is necessary but not sufficient for release-grade scalability claims.

Every distributed result that makes a scalability claim must expose world size,
local world size, node count, rank ownership, memory evidence, communication
evidence, `distribution_semantics`, `claim_evidence_type`,
`scalability_claim_allowed`, and blockers. Plan and preflight summaries should
use `sharding_plan_available=True` instead of treating the plan itself as
release evidence.

## Installation

```bash
pip install -e .
```

For development, install the reproducible test toolchain with
`pip install -e '.[dev]'`. Optional JAX and model examples use
`pip install -e '.[jax]'` and `pip install -e '.[examples]'`, respectively.
Comparison benchmarks may still require PennyLane, provider SDKs, or
accelerator-specific runtimes. Core local examples are designed to run without
distributed initialization.

A Conda environment definition is also provided:

```bash
conda env create -f environment.yml
conda activate flagquantum-dev
```

Verify the install:

```python
import flagquantum as fq

print(fq.__version__)
print(fq.info())
```

## Quick Start

```python
import flagquantum as fq
import torch

theta = torch.tensor(0.3, requires_grad=True)

circuit = fq.Circuit(n_qubits=2)
circuit.h(0)
circuit.cx(0, 1)
circuit.rx(0, theta=theta)

value = circuit.expectation_z(0).sum()
value.backward()

print(float(value.detach()))
print(float(theta.grad))
```

For execution across statevector, MPS, tensor-network, and distributed modes,
use the uniform entry point:

```python
result = fq.run(circuit, mode="auto")

print(result.state)
print(result.plan)
print(result.runtime)
```

`fq.run(...)` returns the stable `fq.ExecutionResult` contract and is the
single recommended execution entry point. `Circuit.run(...)`, `fq.run_native`,
and backend-specific runners such as `fq.run_mps` are compatibility or advanced
interfaces that may expose native result objects.

Training is a separate operation because it updates parameters:

```python
module = fq.Module(build_circuit, n_parameters=2)
optimizer = torch.optim.Adam(module.parameters(), lr=0.01)

training = fq.train(
    module,
    optimizer=optimizer,
    objective=lambda value: value.mean(),
    steps=100,
    log_interval=10,
)
```

Training is silent when `log_interval` is omitted. A positive interval prints
the first step, every matching step, and the final step. Use `callback=` for
per-step integrations such as TensorBoard or experiment tracking; callbacks
run every step independently of terminal logging.

For VQE convergence studies, staged classical and quantum-aware optimization
is available through `flagquantum.algorithms`.
The first implementation supports Adam/AdamW/SGD/L-BFGS, Rotosolve, and exact
full or block-diagonal quantum natural gradient (QNG):

```python
import flagquantum.algorithms as fqa
from flagquantum.algorithms.optimization import OptimizationStage

stages = (
    OptimizationStage("quantum", "adam", steps=40, lr=0.02),
    OptimizationStage(
        "quantum", "qng", steps=20, lr=0.05, block_size=13
    ),
)
result = fqa.run_hybrid_vqe(builder, initial, hamiltonian, stages=stages)
```

QNG currently forms an exact local statevector metric, while Rotosolve assumes
single-frequency Pauli-rotation coordinates. They are convergence tools for
local workloads and do not claim distributed sharded execution. Named groups
passed to `optimize_hybrid` can alternate ordinary classical-network
parameters and quantum-circuit parameters under different stages.

A matched classically gated HVA benchmark compares Adam on both parameter
groups with classical Adam plus layer-block QNG on the quantum group. It
records optimizer steps, circuit evaluations, cumulative wall time, energy,
and relative error, then generates cost-aware PNG/SVG figures:

```bash
python benchmarks/hybrid_classical_quantum_optimizer.py \
  --n-wires 4 --depth 2 --steps 30 --precision float64 \
  --json-output /tmp/hybrid-optimizer.json \
  --csv-output /tmp/hybrid-optimizer.csv
python benchmarks/research/plot_hybrid_classical_quantum_optimizer.py \
  /tmp/hybrid-optimizer.json --output-dir /tmp/hybrid-optimizer-figures
```

This exact-statevector comparison is a local convergence/cost diagnostic, not
distributed-MPS scalability evidence.

For a larger systematic study, the MLP-conditioned suite keeps the classical
optimizer fixed as Adam and compares Adam, block-QNG, L-BFGS, and SPSA on the
quantum parameter group:

```bash
python benchmarks/hybrid_mlp_quantum_optimizer_suite.py \
  --n-wires 6 --depth 3 --steps 60 --precision float64 \
  --json-output /tmp/hybrid-mlp-suite.json \
  --csv-output /tmp/hybrid-mlp-suite.csv
python benchmarks/research/plot_hybrid_mlp_quantum_optimizer_suite.py \
  /tmp/hybrid-mlp-suite.json --output-dir /tmp/hybrid-mlp-suite-figures
```

The MLP consumes layer, gate-family, bond/site-position, and boundary features
and emits a gain and bias for every circuit angle. The output includes
accuracy-target crossing costs and equal circuit-evaluation-budget summaries,
so optimizer-step improvements cannot be mistaken for total-cost improvements.

The experimental A800 Triton block-QNG path uses forward-mode propagation of
the state and all parameter tangents through a complete bond-resolved-phase
Heisenberg HVA. Its benchmark entrypoints are:

```bash
python benchmarks/qng_hva_forward_tangent_triton.py \
  --n-wires 6 --depth 3 --output /tmp/hva-tangent.json
python benchmarks/qng_hva_time_to_accuracy.py \
  --n-wires 4 --depth 2 --steps 60 --output /tmp/time-to-accuracy.json
```

On three N=4/depth2 A800 seeds, reference and Triton block-QNG both passed the
1e-5 gate while Adam passed in zero runs. Triton reduced median time-to-1e-5
from 2.93 s to 1.30 s. This is local complex64 evidence; it is not a
complex128 or distributed-MPS QNG claim.

The deterministic local N=8 Heisenberg convergence check uses complex128,
depth-5 phase-augmented HVA, a dimer-singlet initial state, and Adam-to-L-BFGS:

```bash
python examples/single_machine_quantum_ai/06_heisenberg_hybrid_vqe.py \
  --n-qubits 8 --depth 5 --schedule adam_lbfgs --precision float64
```

It reports `converged` only when relative error against exact diagonalization
is at most `--convergence-tolerance` (default `1e-5`).

The matched 2-GPU distributed-MPS optimizer comparison is available as
lightweight plotting inputs in
`benchmarks/results/heisenberg_vqe/optimizer_comparison_n8_p5_chi16_complex128_g2.{json,csv}`.
The associated SVGs plot optimizer step against energy and relative error.
This convergence path requires saved two-site factorization graphs. Exact MPS
gradient policy retains them automatically; the lower-memory recomputation
pullback is restricted to explicitly approximate-gradient work.

For larger variational circuits, parameters can be organized into named groups:

```python
module = fq.Module(
    build_circuit,
    parameters={"encoder": (4,), "entangler": (3, 2), "readout": ()},
    init={"encoder": "uniform", "entangler": "normal", "readout": 0.1},
    seed=42,
)
```

Passing a symbolic `Circuit` directly also infers and automatically binds its
scalar `fq.Parameter` values during execution.

Use `fq.gate_info("u3")` to inspect a gate's wire count, parameter names, and
parameter shapes without consulting source code.

`fq.run` never performs an optimizer update. `fq.train` runs the PyTorch
optimizer loop and always returns `fq.TrainingResult`.

Plan how the same circuit should run:

```python
plan = circuit.runtime_plan(
    prefer_jax=True,
    state_mode="auto",
    require_gradients=True,
)

print(plan.summary()["recommended_mode"])
print(plan.summary()["recommended_candidate"]["distribution_semantics"])
```

## Local Training Examples

The curated examples intentionally avoid distributed backend initialization:

```bash
python examples/single_machine_quantum_ai/00_local_fast_path_check.py
python examples/single_machine_quantum_ai/01_vqe_statevector.py --steps 2 --n-qubits 3
python examples/single_machine_quantum_ai/03_mps_training.py --steps 2 --n-qubits 4 --max-bond 8
python examples/single_machine_quantum_ai/04_jax_kernel_torch_layer.py --steps 1 --bench-iters 1
python examples/single_machine_quantum_ai/05_mps_1000q_dimer_training.py --steps 2 --n-qubits 20
```

The 1000-qubit MPS example is a structure-aware local MPS workload. It is not
evidence for arbitrary 1000-qubit circuits and does not claim distributed
sharded scalability.

## Deployment Sketch

```python
import flagquantum as fq

circuit = fq.Circuit(2)
circuit.h(0).cx(0, 1)

package = fq.create_deployment_package(
    circuit,
    shots=1024,
)

print(package.backend.provider)
print(package.qasm[:80])
```

Provider integration status depends on credentials, endpoint availability, and
the selected backend profile. See [docs/API.md](docs/API.md) and the deployment
tests for the currently implemented provider abstraction.

## Benchmarks

Install the project, discover maintained scenarios, and run a CPU smoke
benchmark through the stable command:

```bash
pip install -e .
flagquantum-benchmark list
flagquantum-benchmark info statevector_local
flagquantum-benchmark run environment_probe \
  --json-output benchmarks/results/local/environment.json
```

See [benchmarks/README.md](benchmarks/README.md) for local, distributed, and
release-evidence workflows. Script filenames are implementation details; use
`flagquantum-benchmark` in user automation.

## Tests

Local fast-path checks:

```bash
python -m pytest tests/test_native_circuit.py tests/test_backends.py -q
```

Distributed semantics checks:

```bash
python -m pytest tests/test_distributed_statevector.py tests/test_jax_distributed_plan.py tests/test_distributed_scalability_audit.py -q
```

## Contributing

Read [AGENTS.md](AGENTS.md), the
[capability maturity policy](docs/CAPABILITY_MATURITY.md), and the distributed
principle documents before changing distributed runtime, planners, benchmark
claims, or quantum AI training paths.

## License

Apache License 2.0.
