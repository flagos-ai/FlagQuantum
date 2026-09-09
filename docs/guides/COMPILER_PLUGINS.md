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

The normal user journey names the compiler and hardware target directly:

```python
import flagquantum as fq

compiled_ir = fq.compile(
    circuit,
    compiler="qsteed",
    target="quafu:ScQ-P10",
)

from flagquantum.deployment import deploy_circuit
from flagquantum.remote import QuafuProvider

result = deploy_circuit(compiled_ir, QuafuProvider(), shots=1024)
```

The plugin receives the current Quafu chip snapshot, selects a physical
subgraph, and returns logical `CircuitIR` with the ordered physical mapping in
`compiled_ir.metadata["execution_target"]["target_qubits"]`. The circuit still
uses logical wires `0..N-1`. Deployment packaging preserves the compiled
circuit and carries that mapping into Quafu submission without another user
parameter or a second compilation pass.

`deploy_circuit` is the normal remote-execution entry point and creates the
sealed deployment package internally. Call `create_deployment_package`
explicitly only to inspect, persist, sign, or submit that artifact later.

Advanced hosts may pass an explicit target mapping:

```python
from flagquantum.ecosystem.extensions import compile_with_extension

compiled_ir = compile_with_extension(
    source_ir,
    extension="qsteed",
    target={
        "basis_gates": ("h", "x", "rx", "ry", "rz", "cx"),
        "coupling_map": ((0, 1), (1, 2)),
    },
)
```

Missing packages, incompatible SDK versions, capability rejection, compilation
errors, and cleanup failures are reported as extension boundary errors. There is
no implicit compiler selection, substitution, or fallback. Authors and advanced
hosts can use `discover_extensions` and `ExtensionRegistry` directly when they
need explicit lifecycle control.

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
