# CPU vertical slice

Status: implemented on the integration branch.

## Golden path

```text
fq.run(program, options=options)
  -> Runtime Planner asks Compiler to transform the program and builds ExecutionPlan
  -> Runtime validates and executes that exact plan
  -> CPU Platform Provider resolves the requested device
  -> Simulation executes the compiled CircuitIR
  -> Runtime returns ExecutionResult with actual-path diagnostics
```

The path is deliberately limited to local PyTorch statevector execution with
`world_size=1`. It preserves the stable `fq.run`, `fq.plan`, `ExecutionOptions`,
`ExecutionPlan`, and `ExecutionResult` interfaces.

Runtime owns plan validation, device selection, dispatch, and result assembly.
Simulation owns the numerical statevector call. The CPU Platform Provider owns
device availability and identity. Compiler remains the only stage that changes
the program.

`simulation.statevector.local` now owns the local numerical loop and dense Z/Pauli
observables. `Circuit.state()`, `Circuit.expectation_z()`,
`Circuit.expectation_ps()`, and the public `expectation(...)` helper are thin
facades; `Circuit.sample()` retains public format validation while Simulation
owns multinomial and bit conversion numerics. Existing users and Runtime
callers do not change.

### Circuit facade stopping point

The local statevector facade is intentionally complete at this boundary.
`Circuit.probabilities()` remains a one-expression projection of the public
state result; moving it would add only a pass-through helper.
`Circuit.counts()` remains user-facing sampling presentation and format
validation built on `Circuit.sample()`; the random draw and bit conversion
numerics already belong to Simulation. Neither method should move unless a
second concrete consumer needs a shared numerical implementation or an
approved result contract changes their responsibility.

This is a stopping rule, not unfinished migration: do not extract trivial
wrappers merely to make `Circuit` contain no tensor operations. Future work in
this area must remove a real duplicate numerical authority, enable an
implementation replacement, or close a demonstrated cross-domain leak.

Simulation constructs the initial state and owns the algorithms that populate
statevector caches. `Circuit` deliberately retains the cache containers and
mutation-time invalidation: circuit construction, dynamic instructions, and
reusable module binding use one private invalidation entry point. Instruction
changes clear compiled representations and numerical caches; parameter rebinding
clears only the value-dependent state. Moving the containers would spread a new
cache-owner abstraction across Core, Runtime, Simulation, and Benchmarking
without changing product behavior. Revisit this decision only when a second
circuit implementation needs the same lifecycle or the current container
prevents implementation replacement.

The result reports the selected device, platform provider, simulation engine,
and `single_device_fast_path` semantics. An explicit CPU request reports
`cpu_fallback_used=False`; this is a local correctness path, not distributed or
accelerator evidence.

## Run and modify

```bash
python -m examples.cpu_statevector
python -m pytest tests/integration/test_cpu_vertical_slice.py -q
```

The example is the user-facing golden path. The integration test protects the
same journey plus plan-integrity failure behavior.

Start with `flagquantum/runtime/execution.py` for dispatch or result assembly,
`flagquantum/simulation/statevector/local.py` for the numerical entry, and
`flagquantum/providers/platform/pytorch.py` for CPU lifecycle behavior. Changes
to one concern should normally remain in its owning domain.

GPU, distributed execution, noise, MPS, tensor networks, training, QPU, and new
public contracts are outside this slice.

## First physical directory migration

The proven path now resolves its CPU lifecycle through
`flagquantum/providers/platform`. The complete pre-existing platform package
was moved there as one unit so CPU, CUDA, and FlagOS still share one registry
and one set of provider-local contracts. Every in-repository consumer moved in
the same change, and the former `flagquantum/runtime/platforms` package was
deleted rather than retained as a forwarding layer.

The migration reused the existing `PlatformRuntime`, registry, Core capability
values, Runtime plan/result types, and Simulation engine. It added only the
target package marker, concise ownership documentation, and a vertical-slice
test proving that a tampered plan fails before numerical execution. It removed
the old platform directory; the independent extension lifecycle now belongs to
`ecosystem/extensions`.

This is directory ownership evidence, not a new CUDA, FlagOS, domestic-device,
communication, or performance claim. Platform convergence remains in progress
until its separately recorded replacement and hardware-evidence conditions are
met.

## Stable Compiler authority migration

The same proven path now enters `flagquantum/compiler` for canonical
optimization, instruction scheduling, backend lowering, and optional topology
routing. The public `flagquantum.compiler` module became the authoritative
package without changing its expert-facing functions. The former forwarding
file, `compilation/compiler.py`, and `compilation/routing.py` were deleted after
all in-repository consumers switched in the same change.

This migration reused the existing `CircuitIR`, optimization and routing
implementations, public compiler functions, and CPU vertical tests. It added no
new compiler contract, pass framework, registry, fallback, or parallel
implementation. Architecture checks prevent the removed paths from returning.
The disconnected `_compiler` research implementation and its dedicated tests were
removed before release. Compiler replacement is covered by a consumer-facing test
that substitutes the Runtime planning compiler callable while leaving `fq.plan(...)`
and `fq.run(...)` unchanged.

## Runtime planning authority migration

Backend and execution-mode selection, resource estimates, candidate evidence,
topology policy, noisy-backend selection, and backend calibration now live in
`flagquantum/runtime/planner`. The CPU path calls this Runtime-owned planner,
which invokes `flagquantum.compiler` only when program transformation is needed.

This migration reused the existing public planning functions and selection
types, added no manager, registry, compatibility facade, or duplicate policy,
and deleted their former `compilation` modules. Architecture checks prevent the
old paths from returning. The stable `ExecutionPlan` product, serialization,
contract projection, and performance calibration now live under
`flagquantum/runtime`; plan assembly remains in `runtime/planner`. The former
`flagquantum/compilation` transition package has been removed.

## Noise lowering authority migration

Noise-model lowering now lives in `flagquantum/compiler/noise.py`: it performs
the deterministic `CircuitIR + NoiseModel` to channel-bearing `CircuitIR`
transformation. Runtime still selects the noisy execution strategy, and
Simulation still performs channel and trajectory numerics. The former
`flagquantum/compilation/noise` package was deleted rather than retained as a
forwarder.

This move added no new public type or compatibility layer. Existing noisy-plan
types live with the Runtime-owned execution-plan product, and their builder
lives with the Runtime planner. The protected `ExecutionPlan` schema and
identity behavior remain unchanged.

## Calibration policy subtraction

The unused world-size selector was removed before the transition package
exited. Runtime owns world-size, execution selection, and the remaining
performance calibration behind `ExecutionPlan.calibrated_cost()`. Moving or
removing that method still requires a separate protected-contract decision.

## Plan projection consolidation

The single-use `compilation/contract_adapter.py` was folded into Runtime-owned
`execution_plan_contract.py`, which already owns plan identity, serialization,
and contract projection. `ExecutionPlan.to_contract()` retains its signature,
return type, lossy audit semantics, and deterministic identity. Architecture
checks guard the removed adapter and transition package against reintroduction.

## Runtime plan assembly migration

Final stable and noisy plan assembly now lives in `runtime/planner`, beside the
policy decisions it consumes. It remains in the existing planner entry module
so the migration does not spread transitional plan-product dependencies into a
new Runtime file.
Layer reconstruction stays with `execution_plan_contract.py` because it is also
required when restoring the protected serialized plan. The former
`compilation/execution_plan_builder.py` was deleted, and Runtime gained no new
Compiler or transitional-plan dependency edge.

## Planned-program execution

`fq.run(plan)` now restores the compiled or noise-lowered instructions already
carried by `ExecutionPlan.layers` and passes that IR to the executor. The
execution stage no longer repeats noise lowering for stable density-matrix
plans, including plans restored from JSON. Direct `run_native` program entry
points still compile or lower as needed; no serialized field or plan identity
input changed.
