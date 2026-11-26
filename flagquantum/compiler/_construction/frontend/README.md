# Python construction frontend

This package owns source-retaining Python `@qfunc` capture into the existing
checked `HybridProgram` register/effect records. `program.py` coordinates typed
program capture, `python.py` reads source and captures scalar expressions, and
the other modules capture their named source constructs.

Compiler verification, register expansion and candidate selection belong to
`../verification/`, `../codegen/` and `../scheduling/`, respectively. Arithmetic kernels, allocation and gate lowering
are owned by the sibling [`codegen/`](../codegen/README.md) package. This frontend must not execute user bodies, choose runtime backends,
submit jobs or define another IR.

## Dependencies and entry points

Allowed dependencies are the standard library, existing construction semantics
in `../semantics/`, `../analysis/` and `../verification/`, and Compiler's shared `_hybrid` model, diagnostics and
unitary-effect helpers. Sibling frontend modules share capture state and dispatch
through `program.py`. Type-only imports retain the existing cycle boundaries.

The public workflow enters through `flagquantum.synth.qfunc` and
`flagquantum.synth.synthesize`. Internal capture entries are
`program.capture_program` and `python.capture`/`python.capture_named`.
Repository consumers import these authoritative modules directly. Private
`frontend_*.py` aliases were retired as documented in the
[parent migration](../README.md#retired-private-imports).

## Make and verify a small change

For a typed statement change, follow `statements.py` into `program.py` and its
named capture handler. Update an accepted and a rejected scenario in
`tests/team/compiler/test_construction_python_program.py`. Keep effects checked
by `../verification/register_verifier.py`.

Run the existing Bell-state example from the repository root.

```bash
python -m examples.synthesis.synthesis_offline_deployment
```

It captures a two-qubit `@qfunc`, emits OpenQASM 3 with H and CX, and builds an
offline `DeploymentPackage` for `local/offline-line` with 256 shots. It requires
no credentials. Change the source gates in `bell` to exercise another capture
handler. The integration test reads both QASM outputs back and checks the state.

```bash
python -m pytest -q tests/team/compiler/test_construction_python.py tests/team/compiler/test_construction_python_program.py tests/team/compiler/test_construction_frontend_hooks.py
python -m pytest -q tests/integration/test_synthesis_python_program.py tests/integration/test_synthesis_offline_deployment.py
```

For shared Python/Qmod semantics, also run
`tests/team/compiler/test_phase_polynomial_frontends.py` and the relevant
`tests/integration/test_synthesis_qmod_*.py` scenarios.
