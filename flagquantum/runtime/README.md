# Runtime

Turn a quantum program or accepted execution plan into an observable execution
attempt. The design goal is consistent program, gradient, and result semantics
across local, distributed, and remote resources, with a direct local fast path.

Runtime owns selection, execution lifecycle, coordination, recovery, and result
assembly. Compiler transforms programs; Simulation computes numerical updates;
Compute and Remote adapt resources through their contracts.

## Follow the local path

```text
fq.run(program) → plan → execute accepted plan → Simulation → ExecutionResult
```

`fq.run(plan)` must execute the supplied plan without planning or compiling it
again. The result retains that plan and reports the observed execution route.

| Change | Entry point |
| --- | --- |
| Selection and estimates | [planner/](planner/README.md) |
| Plan representation and validation | [execution_plan.py](execution_plan.py), [execution_plan_contract.py](execution_plan_contract.py) |
| Plan execution and dispatch | [plan_execution.py](plan_execution.py), [execution.py](execution.py) |
| Representation lifecycle | [executors/](executors/README.md) |
| Shared rank coordination | [distributed/](distributed/README.md) |
| Stochastic work and recovery | [trajectories/](trajectories/README.md) |
| PyTorch model and optimization loop | [module.py](module.py), [training.py](training.py) |

## Verify a change

From the repository root:

```bash
python -m pytest tests/integration/test_cpu_vertical_slice.py tests/team/runtime -q
python tools/check_architecture.py
python tools/check_dependency_policy.py
```

Preserve the accepted plan, reject unsupported requirements before numerical
execution, and report any authorized fallback. For training changes, verify
parameter updates and recovery equivalence as well as forward values.

[Execution and precision details](IMPLEMENTATION.md) cover dynamic feedback,
precision admission, and internal compatibility paths.
