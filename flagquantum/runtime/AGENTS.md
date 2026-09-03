# Runtime Team Boundary

Runtime owns planning, one execution attempt, resources, sessions, distributed
orchestration, recovery, observability, and evidence assembly. It must not add
Compiler dependencies or numerical algorithms. During migration,
`backends/` belongs to Simulation, `platforms/` to Platform, and
`target_execution.py` to Execution Provider; their nested rules take precedence.
