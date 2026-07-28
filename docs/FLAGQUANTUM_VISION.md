# FlagQuantum Vision

> **Future intent only.** Aspirations here are not current capability or evidence.

## Mission

FlagQuantum aims to become foundational infrastructure for quantum AI:

> Enable researchers and engineers to express quantum-classical models as
> naturally as they use PyTorch today, while allowing the same model and
> training semantics to scale from a laptop to production clusters and
> eventually 10,000-100,000-accelerator leadership systems.

FlagQuantum is not intended to be only a feature-rich quantum simulator. It
should become a coherent platform for constructing, training, compiling,
scaling, validating, and deploying quantum AI workloads.

This document records the long-term expectation. It is a direction and quality
standard, not a claim that the current repository has already achieved it.

## The User Experience We Expect

A user should express a model, objective, accuracy policy, and resource policy
without first learning rank ownership, boundary protocols, contraction slices,
or collective communication.

The intended experience is structurally similar to the following
**non-executable future API sketch**:

```text
import flagquantum as fq
import torch


class Model(fq.Module):
    def __init__(self):
        super().__init__()
        self.quantum = fq.QuantumLayer(
            circuit=build_circuit(),
            observable=fq.PauliSum(...),
        )

    def forward(self, inputs):
        return self.quantum(inputs)


model = Model()
optimizer = torch.optim.Adam(model.parameters())

loss = model(batch).mean()
loss.backward()
optimizer.step()
```

Execution policy should change placement and representation without requiring a
rewrite of model code:

```text
model = fq.compile(
    model,
    target="auto",
    accuracy=fq.AccuracyPolicy(...),
    resources=fq.ResourcePolicy(...),
)
```

The same product model should cover:

- CPU development;
- one accelerator;
- multi-GPU and multi-node execution;
- distributed statevector, MPS, and tensor-network runtimes;
- optional JAX, Triton, cuQuantum, FlagGems, and vendor kernels;
- cloud and QPU deployment after training.

Complexity remains real, but FlagQuantum should manage it through planning,
capability validation, diagnostics, results, and provenance rather than expose
it as mandatory user ceremony.

## PyTorch-Native Foundation

PyTorch-native capability is the mandatory production foundation:

- PyTorch tensors carry primary model data and trainable parameters.
- PyTorch autograd defines the required gradient semantics.
- `torch.optim`, `state_dict`, checkpointing, device movement, and module
  lifecycle behave as PyTorch users expect.
- `torchrun`, `torch.distributed`, Gloo, and NCCL provide the required
  distributed control path.
- DDP, FSDP, and DTensor interoperability is defined where ownership semantics
  are compatible.

JAX and optimized vendor runtimes are important optional execution engines.
They may accelerate quantum kernels, vectorization, compilation, contraction,
or local differentiation, but they must not be prerequisites for core
correctness, training, distributed orchestration, recovery, or deployment.

The intended division is:

```text
PyTorch       product API, autograd, optimization and distributed lifecycle
FlagQuantum   quantum representations, compiler, planner and runtime semantics
JAX/vendors   replaceable high-performance kernel and contraction providers
```

## World-Class SV, MPS, And TN

### Statevector

Statevector should be the exact general reference and the first complete
production distributed path. It should provide:

- a general production gate set;
- native PyTorch forward and reverse mode;
- true amplitude sharding;
- local- and multi-sharded-wire execution;
- communication-avoiding scheduling and gate fusion;
- multi-step sharded optimizer updates;
- checkpoint, recovery, GPU, and multi-node execution;
- rigorous agreement with dense references.

SV will not solve arbitrary thousand-qubit simulation. Its role is to make
exact, highly entangled simulation and training within aggregate cluster memory
correct, fast, and scientifically trustworthy.

### Matrix Product State

MPS should become a defining FlagQuantum capability for structured thousand-
qubit and larger quantum AI:

- variable bond dimensions;
- dynamic site and bond ownership;
- distributed canonicalization and controlled truncation;
- boundary-crossing gate forward and backward;
- shared-parameter reduction and optimizer-state sharding;
- adaptive load balancing and multilevel checkpointing;
- explicit exact or approximate gradient semantics;
- reported discarded weight and error budgets.

FlagQuantum should become a natural platform choice for large, structured,
low-to-moderate-entanglement quantum AI training.

### Tensor Network

TN should become the observable-oriented large-scale engine:

- high-quality contraction planning;
- slicing and subgraph/intermediate sharding;
- distributed task DAG execution and work stealing;
- reverse contraction and environment reuse;
- memory-aware scheduling, retry, and recovery;
- optional high-performance contraction providers;
- thousand-qubit structured observable workloads where path/treewidth permits.

SV, MPS, and TN are complementary representations. A shared product contract
should not erase their distinct ownership, communication, error, and output
semantics.

## An Intelligent, Evidence-Calibrated Planner

The planner should do more than select a backend name. It should explain:

- why SV, MPS, TN, sampling, or deployment is appropriate;
- whether the requested result is exact or approximate;
- expected bond growth, treewidth, slices, and intermediate memory;
- whether a single device will fail for capacity;
- useful accelerator count and topology placement;
- which operations communicate and at which topology tier;
- whether backward and optimizer updates remain sharded;
- expected memory, communication, time, error, and energy;
- cheaper alternatives that satisfy the accuracy requirement.

Planner predictions should be calibrated from measured execution records and
must report uncertainty. Estimates must never be presented as measurements.

## Elegant And Maintainable Infrastructure

FlagQuantum should remain understandable and extensible as it grows. The
expected architecture includes:

- a small, stable public API;
- versioned and validated canonical IR;
- one operator schema and lowering registry;
- enforced package and dependency boundaries;
- immutable, scoped runtime configuration;
- typed and versioned plans, records, measurements, and results;
- strict separation of executor, measurement, provenance, and audit policy;
- no correctness dependence on mutable process-global state;
- no unbounded root namespace or god modules;
- a stable backend/provider/operator extension SDK;
- conformance, compatibility, deprecation, and security policies.

Research flexibility is essential, but experimental implementation must not
silently become stable product behavior. Technical debt that blocks these
properties is production work, not optional cleanup.

## Performance As An Architectural Property

High performance should be designed and measured continuously:

- CPU and single-GPU latency;
- vectorization, gate fusion, allocation reuse, and compile caching;
- mixed precision with explicit numerical policy;
- communication batching and compute/communication overlap;
- topology-aware placement and hierarchical communication;
- strong, weak, throughput, and capacity scaling;
- time and energy to scientific solution.

Leadership-scale ambitions must never justify making local workflows slow or
fragile. Likewise, sophisticated planning and evidence schemas do not replace
efficient kernels and executors.

## Honest Thousand-Qubit Capability

FlagQuantum should pursue:

- thousand-qubit low-bond MPS simulation and training;
- thousand-qubit variable-bond multi-GPU training;
- thousand-qubit TN observable contraction;
- larger structured workloads when computation and communication remain useful.

Every claim must identify representation, circuit topology and depth, maximum
and observed bond dimension, truncation policy, discarded weight, contraction
path/treewidth, requested observable, gradient semantics, precision, hardware,
runtime, and whether the result is exact.

"Supports 1,000 qubits" without these limits is not an acceptable FlagQuantum
claim. General exact thousand-qubit statevector simulation is not a realistic
goal on foreseeable classical hardware.

## From Laptop To 100,000 Accelerators

The long-term runtime must support:

- hierarchical topology and process groups;
- multi-dimensional data/state/site/bond/slice/observable/parameter parallelism;
- decentralized planning and bounded per-rank metadata;
- locality-preserving and communication-avoiding execution;
- multilevel asynchronous checkpointing;
- failure-domain-aware placement;
- localized recovery and elastic replacement where scientifically safe;
- distributed profiling, energy measurement, and provenance.

At 100,000 accelerators, failures and stragglers are normal. No architecture
may assume a flat network, one reliable global coordinator, frequent healthy
global barriers, or complete-state visibility.

Small-scale correctness remains the first step. Every interface and ownership
decision should nevertheless be assessed against whether it can evolve to
hundred-, thousand-, ten-thousand-, and hundred-thousand-accelerator systems
without a framework rewrite.

## Scientific Impact

Accelerator count is not the final objective. FlagQuantum should enable
scientific results that were previously impractical, such as:

- new large-scale quantum learning models;
- trainable representations for strongly correlated systems;
- quantum many-body dynamics;
- high-fidelity quantum material and chemistry observables;
- quantum-classical optimization and representation learning;
- new entanglement-control and compressed-training methods;
- rigorous studies of quantum advantage boundaries.

The highest-level success metric is:

> Because FlagQuantum existed, researchers obtained a trustworthy scientific
> result that could not previously be obtained at the required scale, fidelity,
> or time to solution.

## Gordon Bell Aspiration

FlagQuantum should develop the capability to produce a Gordon Bell Prize-caliber
result. A credible result requires all of:

1. an important scientific problem;
2. original algorithmic innovation;
3. original systems innovation;
4. high sustained performance and scaling efficiency;
5. resilience at the executed scale;
6. scientifically validated outputs;
7. reproducible raw evidence and artifacts;
8. a new scientific conclusion.

Running on many accelerators is not sufficient. Peak theoretical performance,
synthetic FLOP loops, extrapolated scaling, or a planner that can describe a
large topology do not establish this goal.

## Ecosystem Expectation

FlagQuantum should be:

- simple enough for a student completing a first quantum model;
- flexible enough for algorithm and systems research;
- stable enough for production engineering teams;
- efficient enough for leadership supercomputers;
- honest enough for scientific conclusions;
- open enough for an ecosystem of backends, kernels, providers, models, and
  research extensions.

The aspiration is analogous to PyTorch's role in modern AI: not to copy its
implementation, but to achieve a comparable level of usability, composability,
trust, extensibility, and foundational value for quantum AI.

## Final Expectation

FlagQuantum should allow researchers to express the model, objective, accuracy,
and available resources, while the framework responsibly handles
representation selection, compilation, distributed training, error control,
failure recovery, evidence collection, and deployment.

Its full ambition is to serve both ends of the field:

> A student should be able to train a first quantum model on a laptop, and an
> international team should be able to use the same foundational framework to
> execute a previously impossible quantum AI scientific workload on a
> 100,000-accelerator supercomputer.

## Relationship To Product Policy

- `docs/CAPABILITY_MATURITY.md` separates research availability from supported
  and release-certified capability.
- `docs/KNOWN_LIMITATIONS.md` records current support boundaries.
- `docs/EXASCALE_GORDON_BELL_STRATEGY.md` defines leadership-scale system and
  scientific requirements.
- `docs/DISTRIBUTED_QUANTUM_AI_PRINCIPLES.md` and
  `docs/DISTRIBUTED_SCALABILITY_PRINCIPLES.md` define claim and execution
  boundaries.
- Future intent is never proof of current capability; only declared maturity
  and its referenced evidence may support capability claims.
