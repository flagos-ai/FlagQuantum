# Hybrid program compilation

This private Compiler package represents structured quantum-classical programs
with typed SSA values, control-flow regions, and ordered quantum effects.
Its design goal is to lower those programs into the shared Core circuit IR
without losing classical dependencies, parameter bindings, or quantum meaning.

Runtime owns measurement execution and feedback; Simulation owns numerical
evolution. This package does not introduce a second public IR or execution API.

## Follow a program

[Capture](capture.py) → [verify](verifier.py) → [normalize](passes.py) →
[specialize](specialize.py) → [lower](lowering.py).
[Dynamic lowering](dynamic_lowering.py) handles measurement-dependent control.

For a small change, edit the owning stage and add an accepted and rejected
scenario under `tests/hybrid_compiler`. Run from the repository root:

```bash
python -m pytest tests/hybrid_compiler -q
python tools/check_architecture.py
```

Check optimized and unoptimized behavior, including gradients where supported.
Loop expansion must obey its budget; runtime tensor values must not become
part of reusable structure identity.

[Supported constructs and lowering details](IMPLEMENTATION.md) describe the
bounded profiles, parameter flow, and migration constraints.
