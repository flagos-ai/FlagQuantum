# Complete rotation-cost and Beverland formula source closure

All sources are frozen at Qualtran v0.7.0 revision
`bcb17bb25360ed68ce3bc084645db3e929799389`.

| Parent | Complete source | Frozen blob | Native owner |
| --- | --- | --- | --- |
| Q-0925 | rotation_cost_model.py | b747ce18a839b1b5cb0b4b5766c9ba9db1f3fd17 | qec/resources/rotation_synthesis.py |
| Q-0926 | rotation_cost_model_test.py | 2b766a5a36f40f813ce592843278c59383ef26d3 | tests/unit/test_rotation_cost_file.py |
| Q-0904 | beverland_et_al_model.py | 239a262863e381f6d05653398120ee74406ba1df | qec/resources/beverland.py |
| Q-0905 | beverland_et_al_model_test.py | 5da9c101d708bf515e8573a3df46d6feb0c830f6 | tests/unit/test_beverland_model_file.py |

## All rotation definitions

The shared `RotationCostModel` retains both abstract source methods.
`RotationLogarithmicModel` maps to `LogarithmicRotationCost`, retaining slope,
overhead, optional gateset, ceiling order and zero preparation cost.
`ConstantWithOverheadRotationCost` maps to `PhaseGradientRotationCost`, retaining
bitsize and its overhead model. Per rotation it charges max(bitsize - 2, 0)
Toffolis. Preparation charges bitsize copies of the overhead model's rotation
at error_budget / bitsize. The source Beverland and seven-bit presets map to
`BEVERLAND_ROTATION_COST` and `SEVEN_BIT_PHASE_GRADIENT_COST`.

Models retain immutable fields and original logarithm/zero-bit arithmetic
failures. These are analytical gate categories. They do not emit a phase-gradient
state or claim a new executable synthesis protocol. Existing emitters remain
responsible for phases, measurement feedback and clean workspace.

## All Beverland definitions and actual replacement

Source `minimum_time_steps`, `code_distance`, `n_discrete_logical_gates` and
`t_states` map to same-named native functions except
`discrete_logical_gates`. They consume existing `ModeledResources`, with explicit
rotation layers at the established native consumer boundary. All three budget
splits, per-layer preparation, single global preparation in discrete costs and
T-equivalent conversion retain source evaluation order.

The existing `estimate_beverland_resources` now uses these shared functions and
the shared Beverland rotation model. Its signature, defaults, validations,
result fields and zero-step behavior remain unchanged. An independent source
rotation implementation is injected through this real consumer and produces the
same report. The duplicated embedded logarithmic formula is removed.

## Full source comparison and remaining boundaries

All methods are compared for five model configurations and five budgets,
including custom logarithmic coefficients, fixed-width preparation, zero
Toffoli cost at one bit and both presets. Both abstract members and source blobs
are checked. The four Beverland formulas are compared across three synthesis
models, four rotation counts and four layer choices, including zero rotations
and zero explicit layers. Counts, time steps and code distance agree exactly.

The complete original rotation and Beverland tests, including the unchanged
Beverland notebook asset, run against the frozen source. Native notebook
presentation remains a separate Q-0903 item. Existing native input validation
and underflow diagnostics remain at the concrete estimator. New formula-level
methods retain the source arithmetic. Q-0924 strict budget rounding, source
one-pass grid iteration and Q-0613 component-diagram boundaries remain recorded
independently. Primitive emission, QEC demand and physical estimates remain
separate accounting layers.

All 389 concentrated resource regressions and all 14 unchanged source tests
pass, including the original notebook. All four changed Python files pass strict
mypy, Black and Ruff. Core-team scope, architecture, IR and API checks pass.
Q-0925, Q-0926, Q-0904 and Q-0905 close as complete source/test modules.
