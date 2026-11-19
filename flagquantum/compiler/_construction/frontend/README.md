# Python construction frontend

This package owns source-retaining Python `@qfunc` capture into the existing
checked `HybridProgram` register/effect records. `program.py` coordinates typed
program capture, `python.py` reads source and captures scalar expressions, and
the other modules capture their named source constructs.

Compiler verification, register expansion, arithmetic kernels, candidate
selection, allocation and gate lowering remain in the parent construction
package. This frontend must not execute user bodies, choose runtime backends,
submit jobs or define another IR.

## Dependencies and entry points

Allowed dependencies are the standard library, existing construction semantics
in the parent package, and Compiler's shared `_hybrid` model, diagnostics and
unitary-effect helpers. Sibling frontend modules share capture state and dispatch
through `program.py`. Type-only imports retain the existing cycle boundaries.

The public workflow enters through `flagquantum.synth.qfunc` and
`flagquantum.synth.synthesize`. Internal capture entries are
`program.capture_program` and `python.capture`/`python.capture_named`.
The parent `frontend_*.py` modules alias these implementations for existing
consumers, including module-level replacements. Add implementation changes here.
See the [parent mapping](../README.md#python-frontend-implementation) for every
retained import path and its compatibility scope.

## Make and verify a small change

For a typed statement change, follow `statements.py` into `program.py` and its
named capture handler. Update an accepted and a rejected scenario in
`tests/team/compiler/test_construction_python_program.py`. Keep effects checked
by the parent's register verifier.

Run the existing Bell-state example from the repository root.

```bash
python -m examples.synthesis_offline_deployment
```

It captures a two-qubit `@qfunc`, emits OpenQASM 3 with H and CX, and builds an
offline `DeploymentPackage` for `local/offline-line` with 256 shots. It requires
no credentials. Change the source gates in `bell` to exercise another capture
handler. The integration test reads both QASM outputs back and checks the state.

```bash
python -m pytest -q tests/team/compiler/test_construction_python.py tests/team/compiler/test_construction_python_program.py tests/team/compiler/test_construction_frontend_compatibility.py
python -m pytest -q tests/integration/test_synthesis_python_program.py tests/integration/test_synthesis_offline_deployment.py
```

For shared Python/Qmod semantics, also run
`tests/team/compiler/test_phase_polynomial_frontends.py` and the relevant
`tests/integration/test_synthesis_qmod_*.py` scenarios.
