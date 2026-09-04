# Private Compiler domain

## Ownership

`flagquantum._compiler` owns provider-free program transformation: exact import,
verification, analyses, passes, lowering, deterministic code generation, and
Compiler-only target-legality checks. The `compilation` package remains the
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

For target legality, the ten-minute scenario is the narrow-provider rejection
in `tests/internal_ir/test_compiler_target_legality_verdict.py`: a static
requirement needing the full RX parameter domain is rejected when an available
target advertises only a narrower domain.

## Why the legality verdict is a distinct internal type

No existing authoritative type expresses this requirement. Core
`RequirementSet`/`CapabilityMatchResult` intentionally cover the conservative
v1 vocabulary and do not carry legacy gate-parameter coverage, ancilla-policy
ordering, deferred fields, or the original Comparator issues. The legacy
`CapabilityComparison` preserves those issues but has no projection identity,
loss accounting, or required/available semantic fingerprint binding.
`CompilerRequirementProjection` binds the required side only. The frozen
`CompilerLegalityVerdict` is therefore the smallest Compiler-owned attestation
that combines the unchanged legacy comparison with both semantic identities and
loss evidence. It is not a Runtime execution authorization and must not be
promoted to a public API.
