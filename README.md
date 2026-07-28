<div align="center">
  <img src="assets/logo.png" alt="FlagQuantum" width="380">

# FlagQuantum

**The quantum AI framework for FlagOS, unifying programmable training and
scalable execution across classical and quantum hardware.**

[Quick Start](#quick-start) ·
[Capabilities](docs/generated/CAPABILITIES.md) ·
[Examples](examples/README.md) ·
[API Reference](docs/API.md) ·
[Architecture](ARCHITECTURE.md) ·
[Roadmap](docs/FLAGOS_ALIGNED_RELEASE_TRAIN.md)

[![PyTorch](https://img.shields.io/badge/PyTorch-2.5%2B-red.svg)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.10--3.12-blue.svg)](https://python.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

</div>

## One quantum AI program, every execution scale

Quantum AI development is fragmented across local simulators, accelerator
runtimes, distributed systems, tensor representations, and quantum hardware.
FlagQuantum brings these environments under one programming model:

```python
import flagquantum as fq
```

Build and train a quantum or hybrid AI program locally, plan it for the
available resources, scale one logical workload across devices, and package
the trained program for quantum hardware without changing its mathematical
intent.

FlagQuantum is built around four durable ideas:

- **Programmable quantum AI:** circuits, hybrid models, measurements, and
  training share one FlagQuantum-native IR and PyTorch-facing interface.
- **Representation-aware execution:** the same program can use an exact
  statevector, a low-entanglement MPS, a tensor-network path, or a provider
  target according to its workload and constraints.
- **Verifiable multi-chip scale:** state, gradient, optimizer, memory, and
  communication ownership are explicit rather than inferred from process
  count.
- **Classical-to-quantum portability:** training and deployment preserve the
  program, parameter, and result contracts across classical accelerators and
  quantum hardware.

## Quick start

Install the core package:

```bash
pip install -e .
```

Build a differentiable quantum program with the public API:

```python
import flagquantum as fq
import torch

theta = torch.tensor(0.3, requires_grad=True)

program = fq.Circuit(n_qubits=2)
program.h(0)
program.cx(0, 1)
program.rx(0, theta=theta)

value = program.expectation_z(0).sum()
value.backward()

print(float(value.detach()))
print(float(theta.grad))
```

Execute the same program through the uniform runtime:

```python
result = fq.run(program, mode="auto")

print(result.state)
print(result.plan)
print(result.runtime)
```

`fq.run(...)` always returns the stable `fq.ExecutionResult` contract.
Backend-specific runners remain available as advanced interfaces, but
`fq.run(...)` is the recommended execution entry point.

## How it works

```mermaid
flowchart TD
    A["Quantum AI application<br/>fq.Circuit · fq.Module · training loop"]
    B["FlagQuantum Program / IR<br/>operations · parameters · measurements"]
    C["Planning and differentiation<br/>representation · partitioning · gradients"]
    D["Execution runtimes<br/>statevector · MPS · tensor network · provider"]
    E["FlagOS unified multi-chip backend<br/>target integration"]
    F["Classical hardware<br/>CPU · GPU · multi-chip systems"]
    G["Quantum hardware<br/>deployment providers"]
    H["ExecutionResult<br/>values · plans · runtime evidence"]

    A --> B
    B --> C
    C --> D
    D --> F
    D -.-> E
    E --> F
    D --> G
    F --> H
    G --> H
```

FlagQuantum owns quantum program semantics, differentiable training,
representation selection, partitioning, and auditable execution results.
FlagOS is the target unified backend for mapping those plans onto heterogeneous
and multi-chip systems. Quantum providers form a parallel deployment path for
trained programs.

The complete stable `flagquantum.backends.flagos` backend and deeper FlagOS
kernel and collective integration are under active development. Existing
accelerator paths remain supported according to their documented maturity;
the [FlagOS-aligned release train](docs/FLAGOS_ALIGNED_RELEASE_TRAIN.md)
defines when unified backend claims may be promoted.

## Train quantum AI models

`fq.Module` connects parameterized quantum programs to ordinary PyTorch
optimizers:

```python
import flagquantum as fq
import torch

def build_circuit(parameters, inputs=None):
    return (
        fq.Circuit(n_qubits=2)
        .ry(0, theta=parameters[0])
        .cx(0, 1)
        .ry(1, theta=parameters[1])
    )

module = fq.Module(
    build_circuit,
    n_parameters=2,
    policy=fq.RuntimePolicy(observable_wires=(1,)),
)
optimizer = torch.optim.Adam(module.parameters(), lr=0.01)

training = fq.train(
    module,
    optimizer=optimizer,
    objective=lambda value: value.mean(),
    steps=100,
    log_interval=10,
)
```

The same model abstraction supports local quantum AI experiments, hybrid
classical-quantum models, VQE workflows, and distributed training paths.
`fq.run(...)` executes without updating parameters; `fq.train(...)` owns the
optimizer loop and returns `fq.TrainingResult`.

Start with the
[single-machine quantum AI examples](examples/single_machine_quantum_ai/README.md)
or the [tutorial notebooks](examples/tutorials/README.md).

## Plan and scale execution

FlagQuantum keeps execution policy separate from program intent:

```python
plan = program.runtime_plan(
    prefer_jax=True,
    state_mode="auto",
    require_gradients=True,
)

summary = plan.summary()
print(summary["recommended_mode"])
print(summary["recommended_candidate"]["distribution_semantics"])
```

Execution can remain on the local fast path or select a distributed
representation. A production training path must preserve the distribution
semantics through the complete lifecycle:

```text
forward → loss → backward → optimizer update → checkpoint → restart
```

FlagQuantum distinguishes true one-workload sharding from replicated
throughput:

- `single_device_fast_path` is local CPU or one-device execution.
- `data_parallel_replicated`, `rank_local_replicated_kernel`, and
  `replicated_per_rank` do not expand the capacity of one workload.
- `manual_sliced_tensor_contraction` remains a constrained tensor-network
  execution mode until its training and transport blockers are closed.
- `sharded_across_ranks` means one logical workload has explicit rank
  ownership; release claims still require audited hardware evidence.

See the
[distributed quantum AI principles](docs/DISTRIBUTED_QUANTUM_AI_PRINCIPLES.md)
and [distributed scalability principles](docs/DISTRIBUTED_SCALABILITY_PRINCIPLES.md)
for the binding execution and evidence rules.

## Deploy trained programs

The deployment contract packages a trained circuit, its parameters, shots,
provider target, and serialized program:

```python
import flagquantum as fq

program = fq.Circuit(n_qubits=2)
program.h(0).cx(0, 1)

package = fq.create_deployment_package(program, shots=1024)

print(package.backend.provider)
print(package.qasm[:80])
```

Provider availability, credentials, supported operations, and hardware
evidence vary by target. Cloud deployment is currently development evidence,
not a release-certified hardware capability. See the
[API reference](docs/API.md) and [known limitations](docs/KNOWN_LIMITATIONS.md).

## Capability maturity

FlagQuantum assigns maturity to individual capabilities rather than to the
package as a whole:

| Capability | Current level | Current boundary |
| --- | --- | --- |
| Unified circuit API and FlagQuantum IR | `release_certified` | IR v1 changes require an explicit migration |
| Local statevector training | `production_supported` | Capacity is bounded by one device |
| Sharded statevector training | `production_supported` | Multi-node release certification still requires promoted evidence |
| Sharded MPS training | `development_evidence` | Single-node multi-GPU evidence exists; multi-node and release evidence remain incomplete |
| Tensor-network training | `experimental` | General reverse contraction and production transport are not certified |
| Cloud and quantum hardware deployment | `development_evidence` | No provider is release certified |
| Dynamic circuits | `experimental` | Local and provider-preflight scope; no verified QPU execution |
| Extension SDK | `experimental` | Compatibility is not yet guaranteed |

The machine-validated
[capability matrix](capability-maturity.toml) is authoritative. The generated
[capability catalog](docs/generated/CAPABILITIES.md) maps each user goal to its
stable API, runtime mode, gradient support, evidence, examples, and known
boundaries.

## Installation

FlagQuantum supports Python 3.10–3.12 and uses PyTorch as its primary training
interface:

```bash
pip install -e .
```

Install optional environments only when required:

```bash
pip install -e '.[dev]'       # tests, lint, typing, and builds
pip install -e '.[jax]'       # optional JAX quantum kernels
pip install -e '.[cuda]'      # optional Triton kernels
pip install -e '.[examples]'  # model and dataset examples
```

A reproducible Conda environment is also provided:

```bash
conda env create -f environment.yml
conda activate flagquantum-dev
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
| Understand the runtime architecture | [Runtime architecture](docs/RUNTIME_ARCHITECTURE.md) |
| Follow FlagOS integration milestones | [FlagOS-aligned release train](docs/FLAGOS_ALIGNED_RELEASE_TRAIN.md) |
| Inspect limitations and evidence | [Known limitations](docs/KNOWN_LIMITATIONS.md) and [capability maturity](docs/CAPABILITY_MATURITY.md) |

## Benchmarks and evidence

The stable benchmark command discovers maintained scenarios and records
structured results:

```bash
flagquantum-benchmark list
flagquantum-benchmark info statevector_local
flagquantum-benchmark run environment_probe \
  --json-output benchmarks/results/local/environment.json
```

Benchmark conclusions separate local performance, replicated throughput,
manual tensor slicing, and true capacity expansion. Scalability claims are
fail-closed unless runtime ownership, memory, communication, topology,
gradient, optimizer, and blocker evidence pass the release gate.

See [benchmarks/README.md](benchmarks/README.md) for reproducible local,
distributed, and release-evidence workflows.

## Development

Run the smallest meaningful test tier first:

```bash
python tools/ci_tier.py pr-default
```

Runtime, distributed, accelerator, and release changes use progressively
stronger tiers documented in the [testing manual](docs/TESTING.md).

Before contributing, read [AGENTS.md](AGENTS.md), the
[capability maturity policy](docs/CAPABILITY_MATURITY.md), and the
architecture map in [ARCHITECTURE.md](ARCHITECTURE.md).

## License

FlagQuantum is licensed under the [Apache License 2.0](LICENSE).
