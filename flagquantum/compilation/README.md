# Transitional compilation planning

This directory retains the existing execution-plan model, serialization,
assembly helpers, performance-calibration adapter, and noise-lowering/planning
code while those products are separated between Runtime, Core, and Compiler.

Program transformation is authoritative in `flagquantum/compiler`; backend and
execution-mode selection is authoritative in `flagquantum/runtime/planner`.
Do not recreate either implementation here. Start in `models.py` or
`execution_plan_contract.py` only when changing the remaining plan product, and
run the CPU vertical-slice and execution-plan contract tests.
