# FlagQuantum code organization

FlagQuantum uses a layered package model. Dependencies point downward; public
code must not reach sideways into implementation modules.

| Layer | Stable package | Responsibility |
| --- | --- | --- |
| Product API | `flagquantum` | Small, lazy user-facing surface |
| Domain | `flagquantum.core` | IR, parameters, versioned contracts |
| Compilation | `flagquantum.compilation` | Lowering, planning, scheduling |
| Runtime API | `flagquantum.runtime` | Execution, training, configuration |
| Runtime backends | `flagquantum.runtime.backends` | Statevector, MPS, TN, JAX boundaries |
| Simulation primitives | `flagquantum.simulation` | Internal local algorithms, tensor primitives, and compatibility façades |
| Distribution | `flagquantum.runtime.distributed` | Topology, protocols, collectives |
| Governance | `flagquantum.runtime.audit` | Evidence validation and release gates |
| Adapters | `deployment`, `devices`, `extensions` | External systems and plugins |

The v0.1 DTensor device subsystem is a closed internal compatibility island,
not an adapter extension point. Its exact modules are declared in
`architecture.toml`. New imports into that island are forbidden by policy, and
none of its symbols belongs to the v0.2 stable API.

Canonical audit wildcard exports use capability vocabulary. Historical
milestone-numbered names remain available only as lazy explicit-import aliases.
The compatibility-heavy `flagquantum.api` wildcard surface follows the same
rule while preserving explicit attribute access. Its symbol wiring remains in
`flagquantum.api`, while the frozen wildcard manifest is isolated in
`flagquantum._compat_api_exports`; this keeps compatibility behavior auditable
without mixing the two responsibilities.
The canonical audit schema and vocabulary submodules also exclude phase labels
from wildcard exports.

## Import rules

User applications use:

```python
import flagquantum as fq

policy = fq.RuntimePolicy(
    execution_options=fq.ExecutionOptions(mode="mps")
)
```

Framework implementation code may use typed internal boundaries:

```python
from flagquantum.runtime.contracts import DistributedExecutor
from flagquantum.runtime.backends import statevector
from flagquantum.runtime.audit import DistributedEvidenceContract
```

User code must stay on the root `flagquantum` API. Internal runtime,
backend, and audit imports are implementation boundaries and carry no public
compatibility guarantee.

`flagquantum.runtime.backends` is a lazy namespace registry. Inspecting or
importing it does not initialize any backend; requesting one backend package
does not load the others.

`flagquantum.simulation` is not a competing runtime namespace. It owns reusable
local numerical algorithms and dependency-light tensor/kernel primitives.
`flagquantum.runtime.backends` owns backend selection, execution orchestration,
distributed semantics, training records, and evidence. Runtime backends may
consume narrow simulation primitives; simulation modules may not import runtime
or deployment code except for the explicitly registered legacy façades in
`architecture.toml`. Historical `flagquantum.simulation.mps` and
`flagquantum.simulation.tensor` access remains for v1 compatibility, but new
user code stays on the root `flagquantum` API.

The architecture checker rejects reintroduction of the removed compatibility
package or imports from it.

## Module rules

- No wildcard imports in canonical packages.
- Public exports are explicit and typed.
- Optional backends load lazily.
- Domain/core never imports runtime or adapters.
- Executors emit records; audit policy classifies them later.
- Modules default to at most 1,000 lines.
- Every remaining compatibility exception has an owner and removal release in
  `architecture.toml`.

Run `python tools/check_architecture.py` before submitting a change. The checker
rejects new wildcard imports, dependency reversals, reintroduction of removed
runtime paths, unowned exceptions, and module-size regressions.

Runtime return values follow the canonical rules in
`RUNTIME_RESULT_CONTRACT.md`; backend metadata cannot redefine result fields.

## Runtime namespace

`flagquantum.runtime` is the only maintained runtime namespace. Historical
runtime module paths are not shipped and must not be used by source code,
plugins, benchmarks, or serialized artifacts.

## Current architecture state

- Canonical execution, training, configuration, contracts, distributed,
  audit, Statevector, MPS, tensor-network, and JAX boundaries are available.
- Root API, `Circuit`, maintained models, and experimental Statevector exports
  resolve through the canonical runtime.
- `flagquantum.runtime.planner` uses the narrow
  `flagquantum.runtime.planner_adapter` seam and cannot import execution
  implementations.
- `flagquantum.api` remains the frozen v1 compatibility aggregator and resolves
  runtime symbols from canonical modules. Its export manifest is isolated from
  symbol wiring, and the former module-size exception has been removed.
- Reintroducing the removed runtime compatibility package fails architecture
  checks.
- Backend capability registration and runtime backend/dtype configuration have
  completed physical migration. Canonical modules own their single mutable
  registry; the historical modules are identity-preserving façades only.
- The MPS backend is a lazy package boundary under
  `flagquantum.runtime.backends.mps`; checkpoint records, site kernels, and
  production acceptance/planning live in dedicated submodules.
  `mps.production` is canonical and historical production paths are explicit
  compatibility façades.
  Stable MPS exception contracts live in `mps.errors`, allowing planning and
  validation code to remain independent of heavyweight execution engines.
  Workspace-aware factorization policies and microbatch decisions live in
  `mps.factorization`; this module has no dependency on the legacy runtime.
  Critical-path schemas, trace labels, and rank-report aggregation live in
  the dependency-light `mps.profiling` module and are used by benchmarks.
  Distributed gauge normalization lives in `mps.canonicalization` and reaches
  shared tensor transport only through the narrow `mps.communication` adapter.
  Rank-owned state, partition metadata, ownership validation, and pure
  cost-aware ownership planning live in `mps.state`; forward, reverse, and
  training share these contracts directly. Distributed initial-state shape,
  dtype, device, open-boundary, and Bond validation also live there.
  Forward and reverse execution records, results, tapes, and their stable
  runtime summaries share the cohesive `mps.records` boundary, including the
  deterministic Reverse tape-record factory.
  Cross-rank gate exchange, runtime footprint collection, site migration, and
  threshold-based ownership changes share `mps.distribution`, using the narrow
  shared communication adapter.
  Bounded RY/two-site bucket discovery, packing, compiled-kernel execution, and
  factorization recording live in `mps.compiled_layers`, together with its
  cache-lifetime and device-memory observations.
  The rank-owned forward executor itself lives in `mps.forward`; historical
  flat and package paths are explicit compatibility façades.
  Training entry points, lifecycle errors, optimizer ownership, step metrics,
  and final summaries share the cohesive `mps.training` boundary.
  The multi-step lifecycle implementation lives in `mps.training_engine`;
  historical training paths are explicit compatibility façades. Durable
  checkpoint manifests, checksums, writer leases, storage preflight,
  generation retention, save, and restore live in `mps.checkpointing`.
  `mps.training_engine` retains compatibility aliases and is now below the
  default module-size ceiling, so its historical architecture exception has
  been removed.
  Deterministic Reverse SVD validation, VJP segment fusion, gradient bucketing,
  and dirty-Bond planning live in the pure `mps.reverse_planning` module.
  Reverse contracts, records, collective checkpoint accounting, and optional
  factorization-pool budgeting share the cohesive `mps.records` boundary.
  Reverse ownership resolution, initial tensor normalization, default product
  state construction, and runtime state creation live in `mps.state`.
  Tape-segment VJP replay, cross-rank adjoint exchange, parameter accumulation,
  and gradient collectives live in `mps.reverse_replay`.
  Reverse trainable-leaf discovery and per-instruction parameter indexing live
  in `mps.reverse_planning` and are shared directly with Training.
  Reverse observable result contracts, operator-environment transfer,
  single-observable scans and adjoints, Heisenberg MPO energy/adjoint scans,
  and validation of supported term shapes live in `mps.reverse_observables`.
  Z/ZZ fused scans, batch-MSE aggregation, multi-state objective pipelining,
  and observation entry points live in the cohesive
  `mps.reverse_z_observables` algorithm module.
  Reverse orchestration and replay consume shared tensor primitives only
  through the canonical `mps.communication` adapter.
  The state-owning P2P subsystem—including diagnostics, stream ownership,
  buffer pools, descriptor caches, and statistics—lives in
  `runtime.distributed.mps_transport`. MPS has no implementation bridges.
  `mps.communication` is now a stable internal primitive boundary rather than
  a temporary compatibility adapter.
  Transport statistics, cache controls, and communicator warmup are lazily
  exposed from `runtime.distributed`; legacy collectives are direct aliases.
  Descriptor-based, static, packed, and asynchronous P2P tensor paths all share
  the same canonical transport state.
  The canonical transport module has no legacy imports. The canonical
  distributed execution engine is guarded by a reduced 2,350-line ceiling
  while its backend-specific responsibilities are split into focused modules.
  Pure gate-matrix resolution, one/two-site tensor application, splitting, and
  byte accounting already live in `mps.operations`; both canonical and legacy
  executors share those exact function objects.
  Both historical Reverse paths are lazy explicit façades; neither uses
  wildcard re-export.
  The legacy MPS namespace contains no `_impl.py` modules; compatibility paths
  map directly to named canonical modules.
  Sequenced single/batched/static tensor exchange and fixed Reverse metadata
  schemas live in `mps.reverse_transport`, together with asynchronous
  layer-boundary Halo prefetch and CUDA Stream joining.
  Owner-local two-site splitting and thin-QR canonicalization live in the
  cohesive `mps.factorization` module.
  Execution and training composition are isolated in `mps.execution` and
  `mps.training`; neither module depends on legacy runtime implementations.
  The package initializer also contains no legacy implementation dependency.
- The Statevector backend is a lazy package boundary under
  `flagquantum.runtime.backends.statevector`; importing the namespace does not
  initialize forward, reverse, or training engines.
  Its `legacy_device.py` module contains the standalone v0.1 DTensor
  compatibility device. It is not accepted by the Runtime execution path.
  Default distributed execution uses the statevector backend's local
  development simulator or torch-distributed executor. The main Runtime
  execution module contains no device-specific gates;
  `flagquantum.devices` is only the documented compatibility entry and must not
  gain new implementations.
- The Tensor Network backend is a lazy package boundary under
  `flagquantum.runtime.backends.tensor_network`; local simulation and
  distributed execution remain unloaded until their entry point is requested.
  Integer-labelled pair contraction and compiled stage execution live in the
  dependency-light `simulation.tensor_stages` primitive module. Runtime TN
  modules consume that narrow boundary instead of the compatibility-heavy local
  contraction planner. Greedy, multistart, tree-reconfiguration, beam, and
  bounded-optimal path search live in `simulation.tensor_path_search`; the
  historical oversized `simulation.tensor_contraction` exception has been
  removed.
- Provider-neutral deployment contracts remain in `deployment.cloud`.
  Shared HTTP transport lives in `providers.execution.http`, stateless
  counts/QASM parsing lives in `providers.execution.result_parsing`, and the
  Local and remote execution adapters plus provider-specific calibration
  conversion live under `providers.execution`. The
  `deployment.providers` import surface re-exports the same objects and is now
  below the default module-size ceiling, so its size exception has been removed.
- Deterministic remote-style target capability fixtures live in
  `providers.platform.synthetic_remote`; deployment no longer owns platform
  capability producers.
- The optional JAX backend is a lazy package boundary under
  `flagquantum.runtime.backends.jax`; importing the namespace loads neither JAX
  nor its execution adapters.
- Distributed development/production backend policy and LocalTensor simulation
  now physically live in `flagquantum.runtime.distributed.backend_policy`.
  Legacy flat and subpackage paths are explicit identity-preserving façades.
- Backend-neutral distributed request, record, and executor protocols are
  physically owned by `flagquantum.runtime.distributed.protocols`; executor
  adapters and public contracts consume this canonical definition.
- `Module` implementation ownership has moved to
  `flagquantum.runtime.module`; the historical module is a thin façade.
  Consumers use the dedicated `runtime.policy`, `runtime.result`, and
  `runtime.contracts` boundaries rather than the implementation module.
- `RuntimePolicy` and `ExecutionResult` class definitions are now physically
  separated into `runtime.policy` and `runtime.result`. `runtime.module` contains
  only module construction, execution, training, and checkpoint lifecycle.
- Builder bindings, dynamic parameter-slot mappings, compiled-instruction
  records, and detached IR snapshots live in
  `runtime.builder_compilation`. `runtime.module` is below the default 1,000-line
  ceiling and no longer has a size exception.
