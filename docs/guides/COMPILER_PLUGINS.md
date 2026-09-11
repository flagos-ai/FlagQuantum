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

result = fq.run(
    circuit,
    compiler="qsteed",
    target="quafu:Baihua",
    shots=1024,
)
```

The plugin receives the current Quafu chip snapshot, selects a physical
subgraph, and returns logical `CircuitIR` with an ordered physical mapping. The
circuit still uses logical wires `0..N-1`; FlagQuantum carries that mapping
through packaging and submission without another user parameter or compilation
pass. The result remains `fq.ExecutionResult`; access counts with
`result.measurement("counts")` and provider-specific details with
`result.native()`.

Call `fq.compile` separately when the compiled IR must be inspected. Call
`create_deployment_package` only to persist, sign, or submit the sealed artifact
later.

By default QSteed selects the physical subgraph. Expert callers can lock the
ordered logical-to-physical mapping while retaining the same validation and
routing path:

```python
compiled_ir = fq.compile(
    circuit,
    compiler="qsteed",
    target="quafu:Dongling",
    target_qubits=(17, 18),
)
```

The plugin verifies that the physical qubits exist in the current calibration
snapshot and form a connected subgraph. It never substitutes a different
physical mapping when an explicit mapping was requested.

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
