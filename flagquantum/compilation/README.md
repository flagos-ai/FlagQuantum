# Transitional compilation planning

This directory retains the existing execution-plan model, serialization,
assembly helpers, performance-calibration adapter, and noisy execution-plan
product while those products are separated between Runtime and Core.

Program transformation is authoritative in `flagquantum/compiler`; backend and
execution-mode selection is authoritative in `flagquantum/runtime/planner`.
Noise-model lowering is also authoritative in `flagquantum/compiler/noise.py`.
Do not recreate those implementations here. Start in `models.py`,
`execution_plan_builder.py`, or `execution_plan_contract.py` only when changing
the remaining plan product, and run the CPU vertical-slice, noise, and
execution-plan contract tests.

`performance_calibration.py` exists only to preserve the current
`ExecutionPlan.calibrated_cost()` behavior. World-size, backend, and execution
selection calibration belongs in `flagquantum/runtime/planner`; do not add
selection policy to this adapter.
