# Runtime Team Boundary

Runtime owns planning, one execution attempt, resources, sessions, distributed
orchestration, recovery, observability, and evidence assembly. It must not add
numerical algorithms. Compiler dependencies are forbidden except the user-approved
Module frontend capture entry: `module.py` may import `CapturedModuleBuilder` and
`capture_module_builder` from `compiler.module_builder`, replacing its current
builder capture. The contract and replacement fake must land before that
implementation; see `docs/development/MODULE_REUSABLE_EXECUTION_CONTRACT.md`.
The separately user-approved lazy Torch training boundary permits only
`training.py` to import `compile_training_callable` from `compiler.training`.
Its approved contract and replacement fake must land before implementation;
see `docs/development/TORCH_TRAINING_COMPILATION_CONTRACT.md`. It preserves
caller-owned parameters/optimizer state, Torch defaults and checkpoint schemas;
unsupported semantics retain an explicit eager reason. Neither exception permits
compiler imports in executors or Simulation.
`executors/` owns internal,
plan-aware execution lifecycles but delegates numerical kernels to Simulation.
During migration, `target_execution.py` belongs to Remote; its
nested rules take precedence.

The user separately approved the qualified v2 checkpoint writer and strict v1/v2
reader on 2026-11-03. Follow
`docs/development/TRAINING_CHECKPOINT_IDENTITY_CONTRACT.md`; land that contract
and fake before implementation. This permits no additional Compiler import or
Simulation dependency and preserves legacy v1 validation.
