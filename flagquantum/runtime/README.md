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

The experimental IsingBraidProgram uses this same path for the reviewed
four-sigma vacuum sector. `_ising_plan.py` is native admission behind the existing
plan codec, not another plan class or execution manager. Native plans alone use
version 1.1 for braid-only sources or 1.2 for one unread observation, seal matrix outputs and have `analysis=None`; circuit plans
retain version 1.0 and their actual CircuitAnalysis. Dispatch authenticates the
profile from the sealed payload and delegates finite F/R actions to Simulation.
Source snapshots, matrix snapshots and exact environment preflight also apply
to JSON replay. No qubit count or elapsed time is inferred from fusion labels.
The planner and reader bind native plan/source versions exactly, before output
allocation or source decoding. Version 1.2 requires exactly one ordered
`dephase_pair()`; its unread channel may yield a mixed density. No intermediate
outcome, event, classical bit or feedback record is produced. Rehashing cannot
bypass the version or source semantics.

Run `python -m examples.ising_braid_execution` for direct execution, exact JSON
replay and explicit terminal QND charge sampling. It reports plan 1.1, null qubit
count, state shape `(1, 2, 2)`, `P(psi): 0.5`, exact replay and 2,000 terminal
shots without changing caller RNG state. This bounded CPU complex128 route has
no gradients, alternate backend, remote submission, direct shots or circuit
calibration. Each requested matrix expectation retains its own existing
MeasurementResult; no native result wrapper is introduced.

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
