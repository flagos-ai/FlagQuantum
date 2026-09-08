# FlagQuantum code organization

FlagQuantum is organized by responsibility. Dependencies point from orchestration
to domain code and adapters; numerical code never reaches back into Runtime.

| Area | Package | Owns |
| --- | --- | --- |
| Product API | `flagquantum` | Small, explicit user-facing workflow |
| Core | `flagquantum.core` | IR, parameters, capabilities, versioned contracts |
| Compiler | `flagquantum.compiler` | Program optimization, scheduling, routing, lowering |
| Runtime | `flagquantum.runtime` | Planning, execution lifecycle, training, evidence |
| Simulation | `flagquantum.simulation` | Statevector, MPS, tensor-network, noise numerics |
| Execution providers | `flagquantum.providers.execution` | QPU and remote-service adapters |
| Platform providers | `flagquantum.providers.platform` | CPU, CUDA, domestic accelerator, and FlagOS facts |
| Ecosystem | `flagquantum.ecosystem` | Framework conversion, conformance, extension SDK |
| Applications | `flagquantum.algorithms` | Reusable quantum and hybrid algorithms |
| Evaluation | `flagquantum.benchmarking` | Reproducible correctness and performance workloads |
| Agent services | `flagquantum.agent_services` | Deterministic services used by MCP and agents |

`flagquantum.backends` is a thin, stable expert API for backend-native entry
points. Implementations live in `flagquantum.runtime.backends`; external systems
live in `flagquantum.providers`. These packages have similar names but do not
share ownership.

`flagquantum.deployment` remains the maintained user API for sealed deployment
packages and provider-neutral submission. Its concrete service adapters live in
`providers.execution`.

## Dependency direction

The normal execution path is:

```text
User API
  -> Compiler
  -> Runtime planner
  -> Runtime executor
  -> Simulation or execution provider
  -> Result and evidence
```

The important boundaries are:

- Core imports no Runtime, Simulation, Provider, vendor SDK, or framework code.
- Compiler transforms programs; it does not execute them or select devices.
- Runtime chooses resources and owns execution lifecycle; it does not implement
  numerical kernels.
- Simulation receives explicit programs, tensors, dtypes, and algorithm options;
  it does not inspect Runtime plans, process groups, providers, or environment
  policy.
- Providers translate external systems and platform facts; vendor objects stop at
  the provider boundary.
- Ecosystem adapters translate framework objects and delegate execution through
  maintained FlagQuantum APIs.
- Agent and MCP layers call deterministic services; they do not bypass Compiler or
  Runtime contracts.

Cross-boundary data uses explicit contracts. Do not pass internal scheduler,
provider SDK, proof, or cache objects through the public API.

## Where to start

For the smallest CPU workflow, follow:

1. `flagquantum/__init__.py` for the public surface;
2. `flagquantum/compiler/pipeline.py` for optimization;
3. `flagquantum/runtime/planner/` for execution planning;
4. `flagquantum/runtime/execution.py` for execution dispatch;
5. `flagquantum/simulation/statevector/local.py` for local statevector numerics.

Representation-specific JAX numerics live under
`simulation/jax/{statevector,mps,tensor_network}`. The matching Runtime directories
own device placement, sharding, collectives, training lifecycle, and evidence.

Each major package README states what the package owns, what it must not own, and
the shortest modification path. Prefer the narrow owning module over package
facades when changing implementation code.

## Change rules

- A normal feature should primarily change one domain. Repeated four- or five-layer
  edits are an architecture warning.
- Add no new contract type, registry, manager, or extension point without a current
  use case that existing types cannot express.
- Move an implementation and its consumers in one slice. Do not leave a forwarding
  module unless a protected API requires an approved migration.
- Preserve the CPU vertical slice while optional JAX, CUDA, distributed, QPU, and
  service capabilities evolve independently.
- Approximation, precision downgrade, backend substitution, and CPU fallback must
  be explicitly authorized and observable.
- Keep scenario tests that show how the system is used, alongside contract and
  boundary tests.
- Default modules to at most 1,000 lines. Split cohesive internal responsibilities
  without inventing new public concepts.

Run before submitting:

```bash
python tools/check_architecture.py
python tools/check_dependency_policy.py
python tools/check_repository_hygiene.py
pre-commit run --all-files
```

Migration history belongs in `docs/development/team-readiness/` and Git history,
not in this current-state guide.
