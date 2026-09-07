# Operator schema and lowering capabilities

FlagQuantum defines gate and channel semantics once in
`flagquantum.core.operator_schema`. Circuit convenience methods, IR validation,
parameter ordering, drawing, and backend capability checks consume that schema.

`operator_manifest.json` is generated executable documentation. It records each
canonical opcode, aliases, arity, ordered parameters, dtype policy, semantic
kind, adjoint/decomposition metadata, differentiability, wire convention, and
the supported lowering set for every backend.

Regenerate and verify it with:

```bash
python tools/operator_manifest.py
python tools/operator_manifest.py --check
```

Compiler and serializer internals call
`flagquantum.compiler.operator_lowering.validate_lowering` before execution or
serialization. Custom lowering registries use immutable `with_operator` and
`with_capability` updates. FlagQuantum does not expose a custom-gate registry;
custom matrix instructions travel explicitly in Core IR.
