# Shared construction semantics

The Compiler team owns this package. Exact numeric, fixed-word, phase, predicate and unitary semantics shared by syntax consumers belong here.

`numeric_expression.py` owns exact signed-word and binary-scale inference.
`fixed_point.py` and `fixed_point_program.py` own bounded quantized arithmetic.
The phase, predicate, rotation, QPE and unitary modules validate and expand their
named construction effects through existing models and gate helpers.

This package must not capture source syntax, select execution backends, execute
numerical workloads, submit jobs or define another IR. Allowed dependencies are
the standard library, existing Core semantics, Compiler's shared HybridProgram
and construction records, sibling construction implementation modules, and the
existing Torch numerical dependency where required by current validation.

## Entry points and a small change

The public workflow enters through `flagquantum.synth.qfunc` and
`flagquantum.synth.synthesize`. Internal entry points are the existing named helpers consumed by the frontend,
verification and codegen modules.

Start with the named implementation and an accepted/rejected scenario in the
focused tests below. Preserve diagnostic ordering, exact numeric semantics and
cleanup evidence. Run from the repository root with project dependencies.

```bash
python -m pytest -q tests/unit/test_synth_fixed_point.py tests/team/compiler/test_construction_numeric_expression.py
python -m examples.synthesis_offline_deployment
python -m pytest -q tests/integration/test_synthesis_offline_deployment.py
```

The offline Bell-state example captures a Python `@qfunc`, emits OpenQASM 3 with
H and CX, and builds a `DeploymentPackage` for `local/offline-line` with 256 shots.
It requires no credentials. The integration test reads the output back and
checks the Bell state. Change the source gates to exercise the compilation path.
