# Compiler plugins

FlagQuantum compiler plugins are independently installed Python packages. The
main repository owns the `CircuitIR` boundary and plugin lifecycle; the plugin
owns its compiler dependency and translation code. Installing a plugin does not
import or activate it during `import flagquantum`.

## Package registration

A package such as `flagquantum-compiler-qsteed` registers one zero-argument
factory in `pyproject.toml`:

```toml
[project.entry-points."flagquantum.extensions"]
"compiler.qsteed" = "flagquantum_compiler_qsteed:create_extension"
```

The returned extension manifest must use the same identity:

```python
ExtensionManifest(
    name="qsteed",
    version="0.1.0",
    kind="compiler",
    capabilities=frozenset({"circuit_ir"}),
)
```

The extension implements `negotiate`, `start`, `compile`, and `close`.
`compile` accepts a FlagQuantum `CircuitIR`, an optional target mapping, and
returns a FlagQuantum `CircuitIR`. QSteed or QuarkCircuit objects remain inside
the plugin.

## Host use

Discovery is explicit and limited to compiler extensions:

```python
from flagquantum.ecosystem.extensions import (
    CapabilityRequest,
    ExtensionConfig,
    discover_extensions,
)

registry = discover_extensions("compiler")
handle = registry.negotiate(
    "compiler",
    "qsteed",
    CapabilityRequest(required=frozenset({"circuit_ir"})),
)
handle.start(ExtensionConfig())
try:
    compiled_ir = handle.invoke("compile", source_ir, target=target)
finally:
    handle.close()
```

Missing packages, incompatible SDK versions, capability rejection, compilation
errors, and cleanup failures are reported as extension boundary errors. There is
no implicit compiler substitution or fallback.

## Author verification

Before release, a compiler package should run:

```python
from flagquantum.ecosystem.extensions import run_compiler_conformance

report = run_compiler_conformance(create_extension())
```

This initial contract covers circuit-level transpilers. Pulse compilation,
native binaries, and multi-stage MLIR/LLVM artifacts require their own
FlagQuantum-owned artifact contracts and are intentionally not approximated by
generic objects.
