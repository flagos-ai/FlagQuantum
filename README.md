<div align="center">
  <img src="assets/logo_flagquantum.png" alt="FlagQuantum" width="380">

<h1>Make quantum systems learn, scale, and discover.</h1>

<p><strong>FlagQuantum is an open-source distributed differentiable runtime for AI-native quantum science. FlagOS is its target unified backend for heterogeneous accelerators and multi-chip systems.</strong></p>

<p><strong>FlagQuantum makes quantum systems trainable. FlagOS is designed to make their execution portable across accelerators.</strong></p>

<p>Build one trainable PyTorch model, keep one FlagQuantum IR from local development to multi-chip execution, and package the trained program for supported quantum hardware.</p>

[Quick Start](#quick-start) ·
[FlagQuantum × FlagOS](#flagquantum-and-flagos) ·
[Why FlagQuantum](#why-flagquantum) ·
[Scientific Workloads](#scientific-workloads) ·
[Verified Results](#verified-results) ·
[Capabilities](docs/generated/CAPABILITIES.md) ·
[Documentation](docs/reference/API.md)

[![PyTorch](https://img.shields.io/badge/PyTorch-2.5%2B-red.svg)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.10--3.12-blue.svg)](https://python.org/)
[![CI](https://github.com/FlagQuantum/FlagQuantum/actions/workflows/ci.yml/badge.svg)](https://github.com/FlagQuantum/FlagQuantum/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

</div>

## Quick start

Install FlagQuantum and run the maintained one-minute quantum AI example:

```console
pip install -e .
python examples/quick_start.py --mode sv
```

The example trains a classical `torch.nn.Linear` layer and an `fq.Module`
quantum layer in one PyTorch optimizer loop, then checks the result against an
analytical reference. Change only the execution representation:

```console
python examples/quick_start.py --mode mps
python examples/quick_start.py --mode tn
```

At the API level, one program can be trained, executed, and packaged without
changing representations:

```python
import flagquantum as fq
import torch

def build_program(parameters, inputs=None):
    return fq.Circuit(n_qubits=2).ry(0, parameters[0]).cx(0, 1)

model = fq.Module(build_program, n_parameters=1)
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
training = fq.train(
    model,
    optimizer=optimizer,
    objective=lambda value: value.mean(),
    steps=10,
)
trained_program = build_program(next(model.parameters()).detach())
result = fq.run(trained_program)
package = fq.create_deployment_package(trained_program, shots=128)
```

Start with the [annotated quick start](examples/quick_start.py), continue to
the [single-machine quantum AI examples](examples/single_machine_quantum_ai/README.md),
or inspect the stable [`fq.ExecutionResult` contract](docs/reference/RUNTIME_RESULT_CONTRACT.md).

## Beyond single-chip capacity

Distributed capacity and timing records are publishable here only when the
machine-readable capability matrix binds the raw artifact hash, code version,
recorded environment, exact scope, and maturity. See [Verified results](#verified-results)
for the claims that currently pass that fail-closed contract. Other checked-in
benchmark reports remain development records, not public performance claims.

## Why FlagQuantum

### Quantum systems as trainable models

Circuits, measurements, hybrid models, losses, and optimizers participate in
ordinary PyTorch workflows. FlagQuantum treats a quantum program as a model to
differentiate and learn with—not only as a circuit to submit.

### One IR, multiple representations

The same versioned FlagQuantum IR can target an exact statevector, a
low-entanglement MPS, a sliced tensor-network path, or a supported provider.
Runtime planning makes memory, gradient, communication, and fallback
constraints visible before execution.

### Verifiable distributed training

FlagQuantum distinguishes one-workload sharding from replicated throughput.
State, gradient, optimizer, memory, communication, and rank ownership are
reported explicitly. A distributed claim is not inferred from process count.

### From classical accelerators to quantum hardware

Training and deployment preserve the program, parameter, and result contracts.
The goal is one scientific workflow across local development, heterogeneous
multi-chip execution through FlagOS, and supported quantum providers.

## FlagQuantum and FlagOS

**One quantum program across heterogeneous accelerators.**

FlagQuantum and FlagOS form two layers of one quantum scientific computing
stack:

| FlagQuantum defines | FlagOS maps and executes |
| --- | --- |
| `fq.Circuit`, `fq.Module`, and scientific objectives | devices, streams, and memory |
| versioned quantum program semantics | accelerator kernels and compilation |
| PyTorch autograd and quantum gradients | collective communication |
| statevector, MPS, and tensor-network planning | multi-chip topology and placement |
| state, gradient, and optimizer ownership | heterogeneous accelerator adaptation |
| auditable scientific results and deployment packages | execution diagnostics and fallback detection |

The boundary is deliberate: FlagQuantum defines **what** quantum scientific
computation means; FlagOS determines **where and how** it executes efficiently.
The same FlagQuantum program should move from local development to supported
NVIDIA and domestic accelerators without rewriting the scientific model.

FlagOS is a strategic backend, not a prerequisite for getting started. CPU and
single-device paths remain first-class fast paths:

```text
FlagQuantum
    ├── local PyTorch fast path
    ├── optional JAX quantum kernels
    ├── FlagOS heterogeneous multi-chip path
    └── quantum hardware deployment path
```

The stable end-to-end `flagquantum.backends.flagos` integration is under active
development. Cross-accelerator claims are promoted only after native operator
coverage, numerical and gradient parity, hidden-fallback checks, distributed
ownership, and reproducible hardware evidence pass their release gates. See
the [FlagOS-aligned release train](docs/roadmap/FLAGOS_ALIGNED_RELEASE_TRAIN.md).

## Scientific workloads

FlagQuantum connects scientific data, differentiable quantum models, classical
AI optimizers, scalable execution, and quantum hardware through one program
contract.

| Scientific goal | Maintained path |
| --- | --- |
| Ground-state discovery and VQE | [Statevector VQE](examples/single_machine_quantum_ai/01_vqe_statevector.py) and [Heisenberg hybrid VQE](examples/single_machine_quantum_ai/06_heisenberg_hybrid_vqe.py) |
| Hamiltonian identification | [Differentiable MPS workflow](examples/mps_hamiltonian_identification/README.md) |
| Large low-entanglement systems | [1,000-qubit MPS training](examples/single_machine_quantum_ai/05_mps_1000q_dimer_training.py) |
| Representation-aware experiments | [Switch one VQE program across SV, MPS, and TN](examples/vqe_switch_sv_mps_tn.py) |
| Hybrid classical-quantum learning | [Quantum classifier and hybrid examples](examples/single_machine_quantum_ai/README.md) |
| Training-to-hardware workflow | [Parameterized circuit deployment](examples/train_parameterized_circuit_then_deploy.py) |

The long-term direction is quantum scientific intelligence: use gradients and
scientific observations not only to simulate a known quantum system, but to
infer Hamiltonians, optimize quantum dynamics, and discover testable structure.
Current support boundaries remain capability-specific and are listed below.

## How it works

<p align="center">
  <a href="assets/readme/FlagQuantum.png">
    <img
      src="assets/readme/FlagQuantum.png"
      alt="FlagQuantum product architecture across quantum AI applications, unified APIs, IR, planning, classical multi-chip training, and quantum hardware deployment"
      width="1100"
    >
  </a>
</p>

```text
fq.Circuit / fq.Module
        │
        ▼
versioned FlagQuantum IR
        │
        ▼
differentiable runtime planning
        │
        ├── classical execution
        │     ├── local
        │     │     └── statevector / MPS / tensor network
        │     └── distributed
        │           └── sharded statevector / sharded MPS / sliced TN
        │
        └── quantum execution
              └── provider deployment package
```

FlagQuantum owns program semantics, differentiation, representation selection,
partitioning, and auditable execution results. FlagOS maps those plans onto
heterogeneous accelerators and multi-chip systems. Quantum providers form a
parallel deployment path for trained programs.

Local and distributed classical execution use supported PyTorch, JAX, and
FlagOS paths according to their documented maturity. Here, `distributed` is an
architecture category rather than a blanket support claim: sharded
statevector, sharded MPS, and sliced tensor-network execution have distinct
maturity levels in the capability matrix.

The complete stable `flagquantum.backends.flagos` backend and deeper FlagOS
kernel and collective integration are under active development. Existing
accelerator paths remain supported according to their documented maturity.

## Train and plan

`fq.Module` connects a parameterized quantum program to an ordinary PyTorch
optimizer:

```python
import flagquantum as fq
import torch

def build_program(parameters, inputs=None):
    return (
        fq.Circuit(n_qubits=2)
        .ry(0, theta=parameters[0])
        .cx(0, 1)
        .ry(1, theta=parameters[1])
    )

model = fq.Module(
    build_program,
    n_parameters=2,
    policy=fq.RuntimePolicy(observable_wires=(1,)),
)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

training = fq.train(
    model,
    optimizer=optimizer,
    objective=lambda value: value.mean(),
    steps=100,
)
trained_program = build_program(next(model.parameters()).detach())
```

Execution policy remains separate from scientific intent:

```python
program = build_program(torch.tensor([0.3, -0.2], requires_grad=True))
plan = program.runtime_plan(
    prefer_jax=True,
    state_mode="auto",
    require_gradients=True,
)

summary = plan.summary()
print(summary["recommended_mode"])
print(summary["recommended_candidate"]["distribution_semantics"])
```

A production distributed path must preserve its semantics through the complete
training lifecycle:

```text
forward → loss → backward → optimizer update → checkpoint → restart
```

Read the
[distributed quantum AI principles](docs/concepts/DISTRIBUTED_QUANTUM_AI_PRINCIPLES.md)
and [scalability principles](docs/concepts/DISTRIBUTED_SCALABILITY_PRINCIPLES.md)
for the binding execution and evidence rules.

## Deploy to quantum hardware

Package a trained circuit, parameters, shots, provider target, and serialized
program without rebuilding the circuit in another framework:

```python
import flagquantum as fq

program = fq.Circuit(n_qubits=2).h(0).cx(0, 1)
package = fq.create_deployment_package(program, shots=1024)

print(package.backend.provider)
print(package.qasm[:80])
```

Provider availability, credentials, operations, and hardware evidence vary by
target. Deployment is currently development evidence; no provider is release
certified. See the [API reference](docs/reference/API.md) and
[known limitations](docs/reference/KNOWN_LIMITATIONS.md).

## What is ready today

<!-- BEGIN GENERATED CAPABILITY_SUMMARY -->
Maturity applies to each capability—not to the package as a whole.

| Capability | Maturity | Current boundary |
| --- | --- | --- |
| Unified circuit API and FlagQuantum IR | `release_certified` | IR v1; incompatible schema changes require an explicit migration. |
| Local statevector simulation and training | `production_supported` | Capacity is bounded by one device; distributed capacity claims use the sharded capability. |
| Sharded statevector training | `production_supported` | Multi-node release certification remains dependent on promoted audited hardware evidence. |
| FlagOS local statevector CUDA reference | `development_evidence` | CUDA-backed development reference only. It does not certify a domestic accelerator, prove absence of Torch-FL host fallback, establish production performance, or authorize a scalability claim. |
| Differentiable and sharded MPS training | `development_evidence` | Single-node and dual-node execution plus matched checkpoint/restart have development evidence. The only public capacity measurement is emitted from the validated claim below; it is one exact-workload result, not general scalability or release evidence. Boundary instructions still execute serially by owner, and layer-parallel contraction/SVD, capacity multi-step soak, a sealed fault matrix, repeated evidence, and the release payload remain incomplete. |
| Double-Single FP32 numerical primitives | `experimental` | Pure FP32 real and split-complex eager primitives, selective split-statevector P2 reductions, full-state P3, and bounded device-generated-gate P4 experiments are available. P4 removes CPU float64/complex128 gate encoding for its certified gate and angle scope, but optimized kernels, compiled execution, distributed collectives, decompositions, optimizer state, provider-owned Torch-FL route auditing, domestic accelerators, performance, convergence, and production use remain uncertified. Double-Single retains FP32 exponent range and is not generally equivalent to FP64 or complex128. |
| Split real/imag FP32 local statevector P0 | `experimental` | Explicit experimental forward-only executor for a bounded built-in gate set using separate FP32 real and imaginary tensors. It is not selected by the default runtime. Custom matrices, gradients, optimizer steps, sampling and observables APIs, compiled execution, distributed execution, Double-Single storage, provider-internal route auditing, domestic-hardware certification, performance, and production use remain unsupported. CUDA or CUDA-backed flagos evidence is portability evidence only. |
| Split real/imag FP32 observable and parameter-shift P1 | `experimental` | Explicit experimental batch-one Pauli expectation and occurrence-wise two-term parameter-shift gradients for direct named scalar parameters on RX, RY, RZ, RXX, RYY, and RZZ. It is not native autograd and provides no optimizer integration. Parameter expressions, trainable coefficients, custom matrices or states, sampling, compilation, distributed execution, automatic runtime selection, performance, convergence, provider-internal route auditing, and domestic-hardware certification remain unsupported. CUDA or CUDA-backed flagos evidence is portability evidence only. |
| Selective Double-Single split statevector precision P2 | `experimental` | Explicit experimental selective precision path only. State storage, gate generation, and gate application remain split FP32; only Pauli inner products, Hamiltonian term sums, and parameter-shift accumulation retain Double-Single high/low words. Full Double-Single statevectors, native autograd, optimizer integration, residual checkpoints, decomposition, compilation, distributed execution, automatic selection, convergence certification, provider-internal route auditing, domestic-hardware certification, performance, and production use remain unsupported. |
| Full Double-Single split statevector P3 | `experimental` | Explicit correctness-first full Double-Single state experiment. CPU float64/complex128 parameter and gate encoding is required before four FP32 words are transferred to the execution device; state evolution itself has no host fallback. Device-only Double-Single trigonometry, optimized/fused kernels, native autograd, optimizer integration, compilation, distributed execution, automatic selection, algorithmic convergence certification, provider-internal route auditing, domestic-hardware certification, performance, and production use remain unsupported. |
| Device-generated Double-Single split statevector P4 | `experimental` | Explicit correctness-first P4 path for built-in gates and direct scalar parameters within \|angle\| <= 1024. Python scalars are host-ingested as FP32 values; device-resident FP32 or Double-Single parameters remain on device. Parameter expressions, float64 parameter tensors, custom matrices, unbounded angles, native autograd, optimizer integration, compilation, distributed execution, automatic selection, algorithmic convergence certification, provider-internal route auditing, domestic-hardware certification, performance, and production use remain unsupported. |
| CPU PyTorch autograd bridge over Double-Single P5 | `experimental` | Explicit CPU-only first-order PyTorch autograd bridge for direct named scalar FP32 parameters and bounded Pauli expectations. Forward and backward use P4 Double-Single arithmetic internally, but the scalar loss and Tensor.grad are one-word FP32 delivery boundaries and no end-to-end Double-Single gradient claim is made. A separate explicit Double-Single SGD lane is available; higher-order and compiled autograd, CUDA, Torch-FL flagos, FlagCX, distributed execution, automatic selection, convergence, hardware certification, performance, and production use remain unsupported. |
| Single-device precision-preserving Double-Single SGD P5 | `experimental` | Explicit single-device functional high/low master-parameter SGD using P4 high/low parameter-shift gradients. CPU, native CUDA on A800, and CUDA-backed Torch-FL flagos:0 portability trajectories are recorded; the A800 routes do not certify a domestic accelerator or provider internals. It is not torch.optim compatible and has no momentum, weight decay, loss scaling, Adam-family algorithm, checkpoint/state-dict compatibility, higher-order autograd, FlagCX, distributed execution, automatic selection, convergence certification, hardware certification, performance, or production claim. |
| Constrained local MPS TEBD | `experimental` | Static real one-site and adjacent two-site Pauli terms on an open chain, batch one, second-order imaginary-time evolution, and product initial states only. Real-time evolution, periodic and nonlocal terms, gradients, TDVP, distributed execution, and production or scalability claims are unsupported. |
| Tensor-network execution and training | `experimental` | General reverse contraction and production distributed transport are not certified. |
| Exact and trajectory-based noisy simulation | `experimental` | Validated Markovian Kraus channels, timestamped DeviceNoiseProfile input, ASAP gate/idle thermal lowering, classical readout confusion, exact density execution, and reproducible MPS trajectories with single-rank adaptive stopping are available. Pulse overlap, crosstalk, leakage, provider calibration adapters, distributed adaptive stopping, batched statevector trajectories, production multi-GPU scheduling, and noisy gradients remain unsupported. Multi-wire MPS channels use an explicitly dense correctness fallback. |
| Circuit packaging and cloud deployment | `development_evidence` | Provider support and credential/runtime behavior vary; no provider is release-certified by this matrix. |
| Interoperability adapter contract | `experimental` | The adapter API and conformance schema are experimental and currently have two registered implementations, PennyLane and Qiskit. Common conformance proves contract shape, IR round trips, and declared loss handling; it does not install dependencies, sandbox third-party Python, certify provider hardware or numerical equivalence, or permit external objects to enter runtime and accelerator layers. |
| PennyLane QuantumScript interoperability | `experimental` | Certified with PennyLane 0.44.1 and 0.45.1 on Python 3.11 or newer for static QuantumScript conversion and complex128 numerical semantics. QNode, device execution, shots, measurements, trainable parameters, arbitrary wire labels without explicit lossy flattening, and idle wire extents are outside v1. PennyLane objects never enter FlagQuantum runtime, Torch-FL, CUDA, vendor accelerator, or QPU layers. |
| Qiskit IR interoperability | `experimental` | Certified against Qiskit 2.0.x and 2.5.x with Aer 0.17.x through an executable operation, wire-order, statevector, classical-bit, and round-trip contract. Qiskit control flow and arbitrary ParameterExpression import are rejected; named or multiple registers require explicit lossy flattening; custom multi-qubit unitary matrices remain blocked until basis ordering is specified. Conversion does not make Qiskit a runtime dependency or certify any provider hardware. |
| Dynamic circuits and IQM Braket preflight | `experimental` | Provider-neutral conformance vectors pass on the FlagQuantum trajectory runtime and Qiskit Aer. IQM dialect serialization, SDK Program construction, sealed packaging, and mocked provider submission are tested. No AWS account or real IQM QPU task was used, so device availability, published qubit groups, billing, credentials, and hardware results remain unverified. |
| Extension SDK | `experimental` | Extension compatibility is not guaranteed before stabilization. |

The machine-validated [capability matrix](capability-maturity.toml) is the authority.
The generated [capability catalog](docs/generated/CAPABILITIES.md) maps user goals
to APIs, runtime modes, evidence, examples, and known boundaries.
<!-- END GENERATED CAPABILITY_SUMMARY -->

## Verified results

<!-- BEGIN GENERATED PERFORMANCE_CLAIMS -->
Every value below is read from a hash-bound raw artifact. Missing or changed
evidence makes the source-of-truth check fail closed.

| Claim | Maturity and scope | Artifact-derived result | Recorded environment | Evidence identity |
| --- | --- | --- | --- | --- |
| **Sharded MPS exact-workload capacity**<br>`mps-capacity-131072-chi768-20260806` | `development_evidence`<br>One batch-one, complex64, χ768 MPS training step for the checked-in all-rank and all-boundary workload. This is not arbitrary statevector capacity, fixed-plan strong scaling, or release evidence. | **Sites:** 131,072<br>**Logical MPS state:** 1,236,780,012,864 bytes (1,151.84 GiB)<br>**Maximum elapsed time:** 367.37 s<br>**Maximum peak allocated memory per rank:** 77,745,407,488 bytes (72.41 GiB)<br>**Cumulative discarded weight:** 8.39e-06 | **Ranks:** 16<br>**Reported device memory per rank:** 85,093,777,408 bytes (79.25 GiB)<br>**CUDA allocator policy:** expandable_segments:True<br>**Topology fingerprint:** c74a91e3a224a4dd414cfbfbcb31da7570880d42e6aa4eaccecf4b85427940a5<br>**Metadata boundary:** The artifact records rank count, per-rank device memory, topology fingerprint, and CUDA allocator policy. It does not record the exact GPU model or Python, PyTorch, CUDA, NCCL, driver, host, and operating-system versions, so the claim is restricted to the recorded environment fields. | [raw JSON](benchmarks/results/local/mps_capacity_131072q_chi768_16xa800_repeat_complete_20260806.json)<br>SHA-256 `efb0d34c741d1196a7f7c89404cbf033c655f14bf5d575470bbe605bb93b0cbc`<br>code `9d56a6ecd78b06f11b9ee6e8aadcbe9644f2c708` |
<!-- END GENERATED PERFORMANCE_CLAIMS -->

The stable benchmark interface records structured results:

```console
flagquantum-benchmark list
flagquantum-benchmark info statevector_local
flagquantum-benchmark run environment_probe \
  --json-output benchmarks/results/local/environment.json
```

See [benchmarks/README.md](benchmarks/README.md) for reproducible local,
distributed, and release-evidence workflows.

## Installation

FlagQuantum supports Python 3.10–3.12 and uses PyTorch as its primary training
interface:

```console
pip install -e .
```

Optional environments are installed only when needed:

```console
pip install -e '.[dev]'       # tests, lint, typing, and builds
pip install -e '.[jax]'       # optional JAX quantum kernels
pip install -e '.[cuda]'      # optional Triton kernels
pip install -e '.[examples]'  # model and dataset examples
```

Verify the installation:

```python
import flagquantum as fq

print(fq.__version__)
print(fq.info())
```

## Explore FlagQuantum

| Goal | Start here |
| --- | --- |
| Build, compile, or export a program | [`fq.Circuit` quick start](examples/quick_start.py) |
| Train a quantum or hybrid AI model | [Single-machine quantum AI](examples/single_machine_quantum_ai/README.md) |
| Train a large low-entanglement system | [Differentiable MPS](examples/mps_hamiltonian_identification/README.md) |
| Partition one workload across devices | [Distributed statevector](examples/distributed_statevector_topologies/README.md) and [distributed MPS](examples/distributed_mps/README.md) |
| Package a trained program for hardware | [Training-to-deployment example](examples/train_parameterized_circuit_then_deploy.py) |
| Understand runtime architecture | [Architecture](ARCHITECTURE.md) |
| Follow FlagOS integration | [FlagOS-aligned release train](docs/roadmap/FLAGOS_ALIGNED_RELEASE_TRAIN.md) |
| Inspect limitations and evidence | [Known limitations](docs/reference/KNOWN_LIMITATIONS.md) and [capability maturity](docs/roadmap/CAPABILITY_MATURITY.md) |

## Development

Run the smallest meaningful test tier first:

```console
python tools/ci_tier.py pr-default
```

Runtime, distributed, accelerator, and release changes use progressively
stronger tiers documented in the [testing manual](docs/development/TESTING.md).
Before contributing, read [AGENTS.md](AGENTS.md) and
[ARCHITECTURE.md](ARCHITECTURE.md).

## From quantum simulation to quantum scientific intelligence

FlagQuantum is built around a simple idea: quantum systems should become
trainable and scalable components of scientific AI workflows.

Today, that means differentiable programs, representation-aware execution,
verifiable sharded training, and deployment contracts. The long-term goal is a
runtime where scientific data, classical AI, quantum simulation, and quantum
hardware participate in one auditable learning and discovery loop.

## License

FlagQuantum is licensed under the [Apache License 2.0](LICENSE).
