# Runtime Team Boundary

Runtime owns planning, one execution attempt, resources, sessions, distributed
orchestration, recovery, observability, and evidence assembly. It must not add
numerical algorithms. Compiler dependencies are forbidden except the user-approved
Module frontend capture entry: `module.py` may import `CapturedModuleBuilder` and
`capture_module_builder` from `compiler.module_builder`, replacing its current
builder capture. The contract and replacement fake must land before that
implementation; see `docs/development/MODULE_REUSABLE_EXECUTION_CONTRACT.md`.
This exception does not permit compiler imports in executors or Simulation.
`executors/` owns internal,
plan-aware execution lifecycles but delegates numerical kernels to Simulation.
During migration, `target_execution.py` belongs to Remote; its
nested rules take precedence.
