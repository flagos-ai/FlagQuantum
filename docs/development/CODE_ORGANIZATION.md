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
| Remote | `flagquantum.remote` | External control-plane adapters for QPUs and compute services |
| Compute | `flagquantum.compute` | Direct CPU, CUDA, domestic accelerator, and FlagOS access |
| Ecosystem | `flagquantum.ecosystem` | Framework conversion, conformance, extension SDK |
| Applications | `flagquantum.algorithms` | Reusable quantum and hybrid algorithms |
| Evaluation | `flagquantum.benchmarking` | Reproducible correctness and performance workloads |
| Application services | `flagquantum.services` | Reusable capability discovery and composite preflight workflows |

`flagquantum.backends` is a thin, stable expert API for backend-native entry
points. Implementations live in `flagquantum.runtime.executors`; directly
controlled devices live in `flagquantum.compute`; external task systems live in
`flagquantum.remote`.

`flagquantum.deployment` remains the maintained user API for sealed deployment
packages and measurement plans. Provider task handles, submission receipts,
status, and results belong to `flagquantum.remote` with the concrete service
adapters.

## Dependency direction

The normal execution path is:

```text
User API
  -> Compiler
  -> Runtime planner
  -> Runtime executor
  -> Simulation + Compute, or Remote
  -> Result and evidence
```

The important boundaries are:

- Core imports no Runtime, Simulation, Compute, Remote, vendor SDK, or framework code.
- Compiler transforms programs; it does not execute them or select devices.
- Runtime chooses resources and owns execution lifecycle; it does not implement
  numerical kernels.
- Simulation receives explicit programs, tensors, dtypes, and algorithm options;
  it does not inspect Runtime plans, process groups, remote adapters, or environment
  policy.
- Compute isolates directly controlled vendor runtimes; Remote translates external
  task systems. Vendor objects stop at their owning boundary.
- Ecosystem adapters translate framework objects and delegate execution through
  maintained FlagQuantum APIs.
- Application services compose stable APIs only when a multi-step workflow adds
  value. MCP, REST, CLI, and notebook adapters call stable APIs directly for simple
  operations and use Services for shared composite workflows.

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
