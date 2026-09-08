# Runtime Team Boundary

Runtime owns planning, one execution attempt, resources, sessions, distributed
orchestration, recovery, observability, and evidence assembly. It must not add
Compiler dependencies or numerical algorithms. `executors/` owns internal,
plan-aware execution lifecycles but delegates numerical kernels to Simulation.
During migration, `target_execution.py` belongs to Execution Provider; its
nested rules take precedence.
