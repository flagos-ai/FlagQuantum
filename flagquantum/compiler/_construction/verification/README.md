# Construction verification

The Compiler team owns this package. Profile admission, exact operation attributes, register ownership and physical bindings belong here.

`profile.py` owns arithmetic admission and shared diagnostics.
`register_profile.py` owns register types and attribute validation.
`register_verifier.py` owns ordered effects, initialization, aliases and lifetime
checks. `bindings.py` validates physical ports and `constraints.py` validates
whole-program resource ceilings.

This package must not capture source syntax, select execution backends, execute
numerical workloads, submit jobs or define another IR. Allowed dependencies are
the standard library, existing Core semantics, Compiler's shared HybridProgram
and construction records, sibling construction implementation modules, and the
existing Torch numerical dependency where required by current validation.

## Entry points and a small change

The public workflow enters through `flagquantum.synth.qfunc` and
`flagquantum.synth.synthesize`. Internal verification entries are
`profile.verify_construction`, `register_profile.verify_register_program`
and `bindings.verify_bindings`.

Start with the named implementation and an accepted/rejected scenario in the
focused tests below. Preserve diagnostic ordering, exact numeric semantics and
cleanup evidence. Run from the repository root with project dependencies.

```bash
python -m pytest -q tests/team/compiler/test_construction_profile.py tests/team/compiler/test_construction_registers.py
python -m examples.synthesis.synthesis_offline_deployment
python -m pytest -q tests/integration/test_synthesis_offline_deployment.py
```

The offline Bell-state example captures a Python `@qfunc`, emits OpenQASM 3 with
H and CX, and builds a `DeploymentPackage` for `local/offline-line` with 256 shots.
It requires no credentials. The integration test reads the output back and
checks the Bell state. Change the source gates to exercise the compilation path.
