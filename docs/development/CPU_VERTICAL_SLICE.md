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

`simulation.statevector` now owns the local numerical loop. `Circuit.state()`
is a thin public facade, so existing users and Runtime callers do not change.
For this first physical migration, `Circuit` still owns the initial-state and
lifecycle cache containers; moving those containers is separate work and must
not create a second public execution contract.

The result reports the selected device, platform provider, simulation engine,
and `single_device_fast_path` semantics. An explicit CPU request reports
`cpu_fallback_used=False`; this is a local correctness path, not distributed or
accelerator evidence.

## Run and modify

```bash
python -m pytest tests/integration/test_cpu_vertical_slice.py -q
```

Start with `flagquantum/runtime/execution.py` for dispatch or result assembly,
`flagquantum/simulation/statevector.py` for the numerical entry, and
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
the old directory and froze `extensions/sdk` as a separate extension lifecycle
until replacement evidence justifies further convergence.

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
The `_compiler` research implementation remains frozen and off the default path
until a bounded concern can replace the stable implementation under the same
consumer-facing conformance tests.

## Runtime planning authority migration

Backend and execution-mode selection, resource estimates, candidate evidence,
topology policy, noisy-backend selection, and backend calibration now live in
`flagquantum/runtime/planner`. The CPU path calls this Runtime-owned planner,
which invokes `flagquantum.compiler` only when program transformation is needed.

This migration reused the existing public planning functions and selection
types, added no manager, registry, compatibility facade, or duplicate policy,
and deleted their former `compilation` modules. Architecture checks prevent the
old paths from returning. `flagquantum/compilation` temporarily retains the
stable `ExecutionPlan` product, serialization, assembly, contract attachment,
performance calibration, and noise lowering; these are the next bounded seams,
not authorization for new planning policy in that package.
