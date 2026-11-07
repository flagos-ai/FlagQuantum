# Controlled-add and modular resource declarations

The locked Qualtran v0.7.0 commit is
`bcb17bb25360ed68ce3bc084645db3e929799389`. Compiler models describe original
logical resource declarations independently of emitted primitives. Production
code does not import the reference SDK.

## Shared CAdd model (G2-A42)

`arithmetic_resource_model.source_controlled_add_cost(m, n, control_value=...,
legacy_shims=...)` composes the existing predicate and Add models in the source
order: m And computations, one Add and m And-adjoint cleanups. Both control
polarities and both QEC profiles use the same underlying authorities. Unsigned
and Montgomery resource queries at 1 <= m <= n match the original source in
both directions. Symbolic dimensions and signed unequal-width mathematical
semantics are not added by this query.

The root has 1+m+n logical ports. At the last predicate call all earlier loaded
bits are still live. During Add, the original control and input word remain
bystanders. `CostCall` records those lifetimes and yields peak 2m+2n. The model
does not allocate wires or execute a numerical circuit.

The source one-bit Add declaration has -3 Clifford gates; consequently modern
one-bit CAdd has -2. This diagnostic is preserved, not presented as a physical
gate count. Existing emitted one-bit arithmetic remains unchanged. Modern
source QEC counts can omit negative-control X gates while the emitted stream
retains them; the selected profile is explicit.

Run `python -m examples.algorithms.register_controlled_addition` for the existing
append workflow with declared categories/peak and actual primitives printed
separately. It verifies coherent active/inactive branches, absolute phase and
zero work for both polarities. Tests compare all seven source categories and
widths for uint/Montgomery, directions and profiles, replace the shared child
models, and retain the original gate/complex-measurement/cleanup regressions.
