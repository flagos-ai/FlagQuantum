# Hybrid Python capture

This directory owns non-executing Python source capture into the existing
`HybridProgram`. Verification, specialization, optimization and CircuitIR
lowering remain in the parent package. Runtime owns measurement execution and
Simulation owns numerical evolution.

## Implementation ownership

| Module | Responsibility |
| --- | --- |
| `source.py` | Read callable source, select a function and preserve filename and line offsets |
| `diagnostics.py` | Source-located capture diagnostics and errors |
| `program.py` | Capture state, SSA values, ordered effects and statement dispatch |
| `expressions.py` | Scalar expressions, tensor indexing, quantum operands and measurement expressions |
| `control.py` | Branches, loops, region scope restoration and carried bindings |
| `quantum.py` | Quantum statements, observables and terminal results |

These modules depend on the parent hybrid model and verifier plus the Python
standard library. They must not execute user code, inspect runtime tensor values,
select execution backends or depend on provider SDKs. The capture state remains
owned by `program.py`; the other stages operate on that same state. Their type
annotations reference the state through `TYPE_CHECKING` imports.

The existing `flagquantum.compiler._hybrid.capture_source` and
`capture_function` entries call `source.py` directly. The private
`flagquantum.compiler.frontends._structured.capture` module retains aliases to those same
functions and diagnostic types for existing consumers. New capture code imports
its implementation directly. Compiler owns this retained private entry, whose
scope is these four names; remove it when its remaining consumers migrate in an
explicit private-import cleanup targeting the next minor release, 0.5.0. It
introduces no public API or serialized schema.

## Make and test a change

Start with the module for the affected syntax. For example, an index-expression
change belongs in `expressions.py`; a branch-carried value change belongs in
`control.py`. Extend the accepted and rejected scenarios in the capture tests,
then run the existing lowering, forward and gradient paths from the repository
root.

```bash
python -m pytest tests/hybrid_compiler/test_capture.py tests/hybrid_compiler/test_lowering.py tests/hybrid_compiler/test_dynamic_session.py -q
python -m pytest tests/hybrid_compiler/test_forward_vertical.py tests/hybrid_compiler/test_gradient_vertical.py -q
```

For a minimal local workflow, run the accepted-source capture scenario. It
captures and lowers a small program using one tensor argument without importing
an external framework.

```bash
python -m pytest tests/hybrid_compiler/test_capture.py -q
```

Expected result is 25 passing accepted- and rejected-source scenarios.

Preserve operation order, effect ownership, semantic identity, source positions,
error classes and bounded-profile rejection behavior. Unsupported source must
fail before execution. Run `python -m pytest tests/hybrid_compiler -q` and
`python tools/architecture/check_architecture.py` for the broader compiler regression.
