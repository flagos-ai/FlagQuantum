# Private Compiler domain

## Ownership

`flagquantum._compiler` owns provider-free program transformation: exact import,
verification, analyses, passes, lowering, deterministic code generation, and
the existing Compiler target-capability comparison. The `compilation` package remains the
stable/default compilation authority until an approved Core contract and
replacement migration are integrated.

This directory must not own Runtime scheduling or execution lifecycle,
Simulation kernels, device/provider SDK objects, or public API definitions.
Those boundaries are documented in the nearest `AGENTS.md` and the Compiler
convergence inventory.

## Allowed dependencies and entry points

Compiler code may depend on Core IR and capability vocabulary plus other
Compiler-owned implementation modules. It must not import Runtime, concrete
Providers, or numerical engines. The private entry points are intentionally
not public `flagquantum` exports; the default path remains unchanged.

The shortest readable path is the static characterization scenario:

```bash
python -m pytest tests/team/compiler/test_static_pipeline_characterization.py -q
```

Target-capability comparisons remain under the existing
`CompilerRequirementProjection.compare_available()` and
`CapabilityComparison` types. They are not a separate public or cross-domain
legality contract. The first migration candidate is still the static
canonicalization path behind `compilation.compiler.simple_compile`; this README
does not authorize switching the default path.
