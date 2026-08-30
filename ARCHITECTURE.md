# FlagQuantum Architecture

FlagQuantum gives quantum AI programs one public model across local development,
accelerated kernels, distributed simulation, and deployment:

```text
fq.Circuit / fq.Module
          │
          ▼
   FlagQuantum IR
          │
          ├── compile and export
          ├── local statevector, MPS, and tensor-network runtimes
          ├── optional JAX kernels behind the PyTorch interface
          ├── sharded statevector and MPS execution
          └── deployment packages for provider and hardware targets
```

The architectural invariant is simple: backend selection may change execution,
but it must not change the meaning of the program or the result contract.

## Core layers

| Layer | Responsibility | Stable boundary |
| --- | --- | --- |
| User API | Circuit construction, PyTorch modules, planning, execution, training, and deployment | `import flagquantum as fq` |
| FlagQuantum IR | Versioned operators, measurements, metadata, serialization, and validation | `fq.CircuitIR` |
| Planning | Select a representation and execution policy; explain blockers and fallbacks | `fq.plan`, `Circuit.runtime_plan` |
| Runtime | Execute locally or across ranks and return typed evidence | `fq.run`, `fq.ExecutionResult` |
| Training | Preserve PyTorch autograd and optimizer semantics across supported runtimes | `fq.Module`, `fq.train` |
| Deployment | Bind trained parameters, compile for a target, and seal an auditable package | `fq.create_deployment_package` |

Planning is not execution evidence. A plan describes intent and estimates;
runtime records describe what actually ran.

## Source map

```text
flagquantum/
├── api.py                  # curated public API
├── core/                   # circuit IR, gates, devices, and compilation
├── runtime/
│   ├── backends/           # statevector, MPS, tensor-network, and JAX paths
│   ├── distributed/        # ownership, topology, protocols, and execution
│   ├── audit/              # typed evidence, validation, and release gates
│   ├── observability/      # execution and performance records
│   └── *.py                # planning, training, configuration, and results
├── simulation/             # internal local algorithms and numerical primitives
├── deployment/             # target compilation, packages, and providers
├── extensions/             # explicitly experimental extension surface
└── testing/                # reusable correctness and contract helpers

benchmarks/
├── runners/                # reproducible workload and JSON contracts
├── research/               # exploratory analysis and plotting
└── results/                # evidence separated by claim level

contracts/                  # capability and interoperability contracts
tests/                      # unit, integration, distributed, and release gates
docs/                       # current product, architecture, and development truth
tools/                      # repository automation; never a runtime dependency
artifacts/                  # current capability records plus development/legacy classes
paper/                      # isolated manuscript and paper-audit workspaces
```

Detailed subsystem documents live in the
[architecture documentation](docs/architecture/README.md).
Placement and retention rules live in
[repository governance](docs/development/REPOSITORY_GOVERNANCE.md).

## Dependency direction

Dependencies point inward:

```text
API → application services → runtime/compiler → core IR
                            ↘ adapters and providers
```

The core IR does not import runtime backends. Backends consume the IR through
registered lowering contracts. Optional JAX, provider, and hardware integrations
must remain outside the mandatory local PyTorch path.

`simulation` is an internal primitive layer, not a second public runtime.
Runtime backends may reuse its local algorithms and narrow tensor/kernel
primitives, while simulation code cannot depend on runtime orchestration or
deployment except through the legacy façades explicitly registered in
`architecture.toml`. Backend choice and distributed semantics remain owned by
`flagquantum.runtime`.

See the executable
[dependency policy](docs/architecture/ARCHITECTURE_DEPENDENCIES.md) for enforced
package boundaries.

## Execution and training contracts

`fq.run` is the canonical execution entry point. It returns
`fq.ExecutionResult` for supported local and distributed modes. Specialized
native functions are advanced interfaces and may expose backend-specific
objects.

`fq.train` owns the optimizer lifecycle. Distributed training is considered
complete only when forward execution, gradients, optimizer updates, and
checkpoint ownership preserve the declared distribution semantics.

For a claim of distributed scalability, one logical workload must be sharded
across ranks. Replicated data parallelism, rank-local kernels, and manual tensor
slicing are reported under their own semantics and are never relabeled as
capacity expansion.

Read the binding [distributed quantum AI principles](docs/concepts/DISTRIBUTED_QUANTUM_AI_PRINCIPLES.md)
and [scalability principles](docs/concepts/DISTRIBUTED_SCALABILITY_PRINCIPLES.md).

## Public versus internal interfaces

- Public examples use `import flagquantum as fq`.
- Stable names are listed in the
  [generated API inventory](docs/generated/STABLE_API.md).
- `fq.experimental` carries no compatibility guarantee.
- Compatibility modules support migration; they do not define new stable API.
- Benchmark and research utilities must not become runtime dependencies.

Changes to a stable boundary require tests, migration notes, and an explicit
manifest update. Current support levels and known boundaries are published in
the [capability catalog](docs/generated/CAPABILITIES.md).
