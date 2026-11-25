# Architecture Dependency Direction

The implementation separates program transformation, execution orchestration,
and numerical computation:

```mermaid
flowchart TD
    API[Public facade] --> Compiler
    API --> Runtime
    API --> Workflows[Domain workflows]
    Compiler --> Core
    Runtime --> Core
    Runtime --> Simulation
    Runtime --> Kernels[Optimized kernels]
    Runtime --> Compute
    Runtime --> Remote
    Simulation --> Core
    Simulation --> Kernels
    Ecosystem[External object adapters] --> Core
    Collectors --> Records[Execution records]
    Audit[Audit and release policy] --> Records
```

This is an ownership overview, not an exhaustive import graph. The exact forbidden
imports and bounded exceptions are in `architecture.toml` and checked by
`tools/check_architecture.py`.

Reverse edges into IR/core are forbidden. Executors emit backend-neutral
records and cannot import audit/release classification. Importing the stable
root API is lazy and does not load runtime, provider, drawing, distributed, or
optional-kernel modules.

`architecture.toml` owns module-size budgets, import allowlists, and bounded
accelerator-call exceptions. Compatibility exceptions, when introduced, must
name an owner and removal version. Run
`python tools/check_architecture.py` locally and in CI.

## Enforced Package Surfaces

| Responsibility | Package/module |
| --- | --- |
| Stable API | `flagquantum.__init__`, `flagquantum.circuit` |
| Maintained extension APIs | Explicit domain packages such as `flagquantum.compiler`, `flagquantum.noise`, and `flagquantum.deployment` |
| IR and parameters | `flagquantum.core` |
| Compiler | `flagquantum.compiler` |
| Runtime execution planning | `flagquantum.runtime.planner` |
| Runtime API and contracts | `flagquantum.runtime` |
| Realtime messaging, handler calls and channel/session contracts | `flagquantum.realtime` |
| Distributed executor protocols | `flagquantum.runtime.distributed.protocols` |
| Backend boundaries | `flagquantum.runtime.executors` |
| Distributed orchestration | `flagquantum.runtime.distributed` |
| Numerical semantics and portable references | `flagquantum.simulation`, including optional `simulation.jax` |
| Optional optimized kernels | `flagquantum.kernels`, including lazy `kernels.triton` |
| Directly controlled resources | `flagquantum.compute` |
| Measurement contracts | `flagquantum.core.ir`, `flagquantum.runtime.measurements` |
| Deployment packages | `flagquantum.deployment` |
| External compute and QPU providers | `flagquantum.remote.compute`, `flagquantum.remote.qpu` |
| External framework adapter contract | `flagquantum.ecosystem` |
| External framework conversion | `flagquantum.ecosystem.<framework>` |
| Evidence and audit policy | `flagquantum.runtime.audit` |

`flagquantum.runtime` is the sole runtime implementation namespace. New
application, plugin, and backend code must use its explicit contracts,
executors, distributed, and audit subpackages.

`flagquantum.realtime` is a separate domain for the software messaging boundary.
It imports only itself and the standard library. Core, Compiler, Runtime,
Simulation, Kernels, QEC and Services cannot import either realtime path without
a separately approved integration. `flagquantum.remote.realtime` contains only
import compatibility; it owns no framing, decoder, transport or execution code.
[ARCH-014](decisions/ARCH_014_REALTIME_PACKAGE_OWNERSHIP.md) records this migration.

The circuit implementation lives at `flagquantum.circuit`; `flagquantum.core`
contains backend-neutral IR, operator schemas, parameters, configuration, and
versioned contracts only.

External framework objects stop at `flagquantum.ecosystem`. Its immutable lazy
registry and framework-neutral conversion contracts are the shared control-plane
boundary; concrete implementations live under `flagquantum.ecosystem.<framework>`.
The Qiskit and PennyLane adapters convert to or from versioned FlagQuantum IR,
report semantic loss explicitly, and load their external framework only when
conversion is called. PennyLane v1 stops specifically at immutable
`QuantumScript`; QNodes, devices, execution, and autograd remain outside it.
Runtime kernels, distributed worker contracts, CUDA, and FlagOS never receive
external framework objects.
