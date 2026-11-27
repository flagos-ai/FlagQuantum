# Construction candidate scheduling

The Compiler team owns this package. Construction schedule selection, budgets, resource comparison and target constraints belong here.

`selection.py` compares existing arithmetic and register candidates. `target.py`
checks supported target constraints and constructs target compilation evidence.

This package must not capture source syntax, select execution backends, execute
numerical workloads, submit jobs or define another IR. Allowed dependencies are
the standard library, existing Core semantics, Compiler's shared HybridProgram
and construction records, sibling construction implementation modules, and the
existing Torch numerical dependency where required by current validation.

## Entry points and a small change

The public workflow enters through `flagquantum.synth.qfunc` and
`flagquantum.synth.synthesize`. Internal scheduling entries are
`selection.select_arithmetic`, `selection.select_register_program` and
`target.compile_target`.

Start with the named implementation and an accepted/rejected scenario in the
focused tests below. Preserve diagnostic ordering, exact numeric semantics and
cleanup evidence. Run from the repository root with project dependencies.

```bash
python -m pytest -q tests/team/compiler/test_construction_selection.py tests/team/compiler/test_construction_selection_expression.py
python -m examples.synthesis.synthesis_offline_deployment
python -m pytest -q tests/integration/test_synthesis_offline_deployment.py
```

The offline Bell-state example captures a Python `@qfunc`, emits OpenQASM 3 with
H and CX, and builds a `DeploymentPackage` for `local/offline-line` with 256 shots.
It requires no credentials. The integration test reads the output back and
checks the Bell state. Change the source gates to exercise the compilation path.
