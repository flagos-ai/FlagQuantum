# Construction analysis

The Compiler team owns this package. Expression rewrites, range and bit facts, and arithmetic use/lifetime analysis belong here.

`arithmetic_rewrites.py` rewrites admitted arithmetic expressions. `ranges.py`
computes integer widths, `bit_facts.py` propagates known bits, and `lifetimes.py`
computes dependencies and uses for scheduling.

This package must not capture source syntax, select execution backends, execute
numerical workloads, submit jobs or define another IR. Allowed dependencies are
the standard library, existing Core semantics, Compiler's shared HybridProgram
and construction records, sibling construction implementation modules, and the
existing Torch numerical dependency where required by current validation.

## Entry points and a small change

The public workflow enters through `flagquantum.synth.qfunc` and
`flagquantum.synth.synthesize`. Internal analysis entries are
`arithmetic_rewrites.normalize_arithmetic` and `lifetimes.analyze_lifetimes`.

Start with the named implementation and an accepted/rejected scenario in the
focused tests below. Preserve diagnostic ordering, exact numeric semantics and
cleanup evidence. Run from the repository root with project dependencies.

```bash
python -m pytest -q tests/team/compiler/test_construction_semantics.py tests/team/compiler/test_construction_scheduling.py
python -m examples.synthesis_offline_deployment
python -m pytest -q tests/integration/test_synthesis_offline_deployment.py
```

The offline Bell-state example captures a Python `@qfunc`, emits OpenQASM 3 with
H and CX, and builds a `DeploymentPackage` for `local/offline-line` with 256 shots.
It requires no credentials. The integration test reads the output back and
checks the Bell state. Change the source gates to exercise the compilation path.
