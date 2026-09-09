# Runtime

Runtime turns an accepted program or execution plan into one observable
execution attempt. It owns backend selection, execution lifecycle, fallback
policy, distributed coordination, checkpoint/restart orchestration, and result
assembly.

Runtime does not implement compiler transformations, numerical simulation
kernels, vendor SDK adapters, or long-lived service scheduling. Those belong to
Compiler, Simulation, Compute, and Remote respectively.

The experimental dynamic executor may orchestrate a validated `NoiseModel`.
It owns random-stream use, true-versus-observed measurement flow, conditional
control from observed bits, event accounting, and result assembly. Numerical
bit-flip and readout sampling kernels remain in Simulation. The current dynamic
profile rejects general Kraus channels, correlated readout, device-timing
profiles, and noisy gradients.

A private bounded feedback plan may declare measurement decision points for
the local trajectory path. Runtime validates those points, invokes a stateless
controller with the shot's observation history, applies bounded physical-X or
Pauli-frame-X actions, and records true bits, observed bits, decisions, and
frame evolution. This is synchronous local development evidence; it is not a
stable plugin or hard-real-time/provider controller contract. Explicit batched
feedback fails closed.

## Local CPU path

The shortest supported path is:

```text
fq.run(program)
  -> runtime.execution.run()
  -> runtime.planner.plan()
  -> runtime.plan_execution.execute_plan()
  -> runtime.execution.run_native()
  -> simulation statevector kernel
  -> ExecutionResult
```

Start in `planner/` for selection policy, `execution_plan.py` for the immutable
plan product, `execution_plan_contract.py` for serialization and validation,
`plan_execution.py` for exact-plan execution, and `execution.py` for local
dispatch. Backend-specific lifecycle code belongs in `backends/`; numerical
tensor operations do not.

For program input, `fq.run(program)` invokes the stable planner once; that
planning step compiles once and seals a canonical executable program into the
plan. Execution passes the same compiled instructions and numerical semantics
to Simulation for one numerical launch; plan-only metadata need not enter the
numerical layer.

Executing an existing plan is a strict suffix of this path: `fq.run(plan)`
starts at `execute_plan()`, does not invoke planning, mode selection, compilation,
or plan construction again, and launches the selected Simulation entry exactly
once. The returned `ExecutionResult` retains the supplied plan object.

`run_advanced()` remains an internal characterization bridge for backend-specific
tests. It is not part of Runtime's declared exports or the experimental public
namespace. Do not add product callers; use `fq.run()` for stable execution or an
owned backend facade for backend-native results.

`records.py` constructs Core-owned execution records from observed facts. It
must not define a second record schema or make release-eligibility decisions.

Target selection accepts a precision route only when native, effective,
storage, and software-mechanism facts are observed. If those facts describe
software-expanded precision, the effective dtype also requires
certification-level evidence whose snapshot scope names that dtype. This makes
the mechanism selectable only inside its certified numerical scope; it does
not promote an experimental implementation to stable `complex128` support.
`native_dtype` and `storage_dtype` name scalar lanes (`float32` or `float64`),
while `effective_dtype` names logical complex state precision (`complex64` or
`complex128`). Runtime rejects mixed vocabulary instead of treating `float64`
and `complex128` as aliases.

## Ten-minute change path

For a small local execution change, modify one Runtime owner and run:

```bash
python -m pytest tests/integration/test_cpu_vertical_slice.py \
  tests/team/runtime -q
python tools/check_architecture.py
python tools/check_dependency_policy.py
```

Keep ordinary changes inside Runtime. If a change also requires Compiler
lowering or Simulation math, split it at the existing program/plan or numerical
kernel boundary before implementation.
