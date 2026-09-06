# Runtime

Runtime turns an accepted program or execution plan into one observable
execution attempt. It owns backend selection, execution lifecycle, fallback
policy, distributed coordination, checkpoint/restart orchestration, and result
assembly.

Runtime does not implement compiler transformations, numerical simulation
kernels, vendor SDK adapters, or long-lived service scheduling. Those belong to
Compiler, Simulation, Providers, and external services respectively.

## Local CPU path

The shortest supported path is:

```text
fq.run(program)
  -> runtime.execution.run()
  -> runtime.planner.plan()
  -> runtime.plan_execution.execute_plan()
  -> runtime.execution.run_native()
  -> simulation statevector kernel
  -> ExecutionResult
```

Start in `planner/` for selection policy, `plan_execution.py` for exact-plan
validation, and `execution.py` for local dispatch. Backend-specific lifecycle
code belongs in `backends/`; numerical tensor operations do not.

For program input, `fq.run(program)` invokes the stable planner once; that
planning step compiles once and seals a canonical executable program into the
plan. Execution passes the same compiled instructions and numerical semantics
to Simulation for one numerical launch; plan-only metadata need not enter the
numerical layer.

Executing an existing plan is a strict suffix of this path: `fq.run(plan)`
starts at `execute_plan()`, does not invoke planning, mode selection, compilation,
or plan construction again, and launches the selected Simulation entry exactly
once. The returned `ExecutionResult` retains the supplied plan object.

`records.py` constructs Core-owned execution records from observed facts. It
must not define a second record schema or make release-eligibility decisions.

## Ten-minute change path

For a small local execution change, modify one Runtime owner and run:

```bash
python -m pytest tests/integration/test_cpu_vertical_slice.py \
  tests/team/runtime -q
python tools/check_architecture.py
python tools/check_dependency_policy.py
```

Keep ordinary changes inside Runtime. If a change also requires Compiler
lowering or Simulation math, split it at the existing program/plan or numerical
kernel boundary before implementation.
