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
| Deployment | Bind trained parameters, compile for a target, and seal an auditable package | `flagquantum.deployment.create_deployment_package` |

Planning is not execution evidence. A plan describes intent and estimates;
runtime records describe what actually ran.

## Source map

```text
flagquantum/
├── __init__.py             # lazy stable `fq` facade
├── _api.py                 # root compile, plan, and run composition
├── circuit.py              # Circuit construction
├── core/                   # backend-neutral IR and shared semantics
├── compiler/               # validation, optimization, lowering, and code generation
├── runtime/                # planning, execution lifecycle, results, and coordination
├── simulation/             # numerical methods and kernels
├── noise/                  # backend-neutral noise models and channels
├── observables/            # user-facing measurement construction
├── qec/                    # error-correction workflows and domain models
├── twin/                   # hardware digital-twin models
├── compute/                # resources controlled by the current process
├── remote/                 # external task systems and result retrieval
├── ecosystem/              # framework and format adapters
├── deployment/             # sealed target-neutral execution packages
├── services/               # reusable multi-step application workflows
├── algorithms/             # user-facing algorithm composition
├── benchmarking/           # reproducible measurement and evidence generation
├── drawer/                 # circuit visualization
├── testing/                # reusable correctness and conformance helpers
└── experimental/           # explicitly unstable APIs

benchmarks/
├── runners/                # reproducible workload and JSON contracts
├── research/               # exploratory analysis and plotting
└── results/                # evidence separated by claim level

contracts/                  # capability and interoperability contracts
tests/                      # unit, integration, distributed, and release gates
docs/                       # current product, architecture, and development truth
tools/                      # repository automation; never a runtime dependency
artifacts/                  # current capability records plus bounded development output
```

Detailed subsystem documents live in the
[architecture documentation](docs/architecture/README.md).
Placement and retention rules live in
[repository governance](docs/development/REPOSITORY_GOVERNANCE.md).

## Dependency direction

Dependencies point inward:

```text
User facade → Compiler / Runtime / application workflows → Core
                         │
                         ├── Simulation numerical methods
                         ├── Compute adapters for local resources
                         └── Remote adapters for external task systems
```

Core does not import orchestration, numerical engines, or vendor integrations.
Compiler transforms programs but does not execute them. Runtime organizes
execution but does not implement numerical kernels. Simulation may consume Core
semantics but does not select resources. Compute and Remote isolate hardware and
external-system details from the other domains. Optional integrations must remain
outside the mandatory local PyTorch path.

`simulation` is a numerical-method domain, not a second public runtime. Runtime
may call its engines through explicit entry points, while backend choice,
distributed ownership, recovery, and evidence assembly remain Runtime concerns.

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

### Package-root ownership

The Python files directly under `flagquantum/` are a closed public-facade set;
new implementations belong in their owning domain rather than at package root.

| File | Responsibility |
| --- | --- |
| `__init__.py` | Lazily exposes the reviewed `fq` surface. |
| `_api.py` | Composes the root `compile`, `plan`, and `run` journeys. |
| `circuit.py` | Owns circuit construction and circuit-facing convenience methods. |
| `dynamic.py` | Preserves the reviewed dynamic-circuit namespace. |
| `errors.py` | Owns stable cross-domain error categories. |
| `gradients.py` | Owns backend-neutral user gradient utilities. |
| `models.py` | Owns maintained user-facing hybrid model examples. |
| `operators.py` | Preserves the reviewed operator discovery namespace. |
| `training.py` | Preserves the reviewed training lifecycle namespace. |
| `version.py` | Owns package version discovery. |

The three small namespace files are intentional compatibility boundaries, not
places for new implementation logic. Removing or renaming them requires the
public API process; their size alone is not evidence of redundancy.

Changes to a stable boundary require tests, migration notes, and an explicit
manifest update. Current support levels and known boundaries are published in
the [capability catalog](docs/generated/CAPABILITIES.md).
