# Transitional compilation planning

This directory retains the existing execution-plan model, serialization,
performance-calibration adapter, and noisy execution-plan product while those
products are separated between Runtime and Core.

Program transformation is authoritative in `flagquantum/compiler`; backend and
execution-mode selection is authoritative in `flagquantum/runtime/planner`.
Noise-model lowering is also authoritative in `flagquantum/compiler/noise.py`.
Do not recreate those implementations here. Runtime plan assembly lives in
`flagquantum/runtime/planner`. Start in `models.py` or
`execution_plan_contract.py` only when changing the remaining plan product, and
run the CPU vertical-slice, noise, and execution-plan contract tests.

`performance_calibration.py` exists only to preserve the current
`ExecutionPlan.calibrated_cost()` behavior. World-size, backend, and execution
selection calibration belongs in `flagquantum/runtime/planner`; do not add
selection policy to this adapter.

## Current stopping point

The package contains only `models.py`, `execution_plan_contract.py`, and
`performance_calibration.py`. A consumer audit found no unused source module
that can be deleted independently:

- `models.py` owns the protected `ExecutionPlan` and transitional noisy-plan
  products;
- `execution_plan_contract.py` owns protected serialization, identity,
  validation, and structural reconstruction;
- `performance_calibration.py` is the sole implementation behind the existing
  `ExecutionPlan.calibrated_cost()` method.

Do not move, inline, or delete these files solely to empty this directory.
Relocating the plan product or its qualified types requires the approved public
API migration, serialization fixtures, compatibility analysis, and consumer
conformance. Until then, this package is frozen: it may receive contract fixes,
but no new Compiler or Runtime policy.
