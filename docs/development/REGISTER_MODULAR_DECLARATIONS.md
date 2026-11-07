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

## Modular placeholder declarations (G2-MOD15)

`modular_declarations.py` ports the four direct call declarations in the original
`arithmetic/_shims.py`. It uses existing flat register layouts, typed resource
callees and `SymbolicResourceFormula`, without a new model registry or SDK type.
This slice accepts concrete nonnegative widths. Symbolic dimensions remain a
separate open path; negative widths are invalid native register dimensions.

| Native declaration | Original source | Direct calls |
| --- | --- | --- |
| `multi_controlled_x` | MultiCToffoli | n-2 Toffoli |
| `controlled_subtraction` | CSub | CAdd plus three BitwiseNot |
| `less_than` | Lt | n Toffoli |
| `controlled_half` | CHalf | n TwoBitCSwap |

`modular_arithmetic_registers` preserves the source names and flat widths.
`modular_arithmetic_calls` returns the literal direct multiplicities.
`source_modular_arithmetic_resource_formula` consumes the existing typed QEC
query and original signature/callee width fallback. CSub uses the shared CAdd
composition in the chosen profile; it does not duplicate predicate/Add math.
Lt's `signed` flag changes neither its original signature nor its declaration.

Zero call weights remain visible. The original MultiCToffoli with zero/one
controls declares -2/-1 Toffoli; those diagnostics remain declaration data and
are not passed off as executable repetition counts. The original width fallback
also includes the three-bit atomic callee at zero/negative multiplicity. Native
executable `CostCall` validation is unchanged. Zero-width CSub has failing source
QEC and width queries and is rejected explicitly; its register metadata can
still be inspected.

All four original placeholders lack gate and classical decompositions. There
is therefore no source gate stream, complex-amplitude channel or measurement
history to compare for them. The native output is a formula, not an appendable
operation. The actual shared CAdd emitter continues to carry its independent
gate/state/clean-work evidence. The runnable example
`python -m examples.algorithms.modular_resource_declarations` prints declared categories,
peak and T equivalents; it makes no physical-gate or quantum-execution claim.
Tests compare original signatures, direct callees/multiplicities, every QEC
category, both profiles/directions, the signed flag and source failure boundaries.
