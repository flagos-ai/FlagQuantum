# Classical simulation notebook

Reference: unchanged Qualtran v0.7.0 commit
`bcb17bb25360ed68ce3bc084645db3e929799389`,
`qualtran/simulation/classical_sim.ipynb`, blob
`9f99d7e8d86996c40fddf84c275993571a9ed883`.
The native asset is `examples/notebooks/classical_simulation.ipynb`; its small
composition helper is `examples/algorithms/classical_simulation.py`. Both retain
source attribution. Run the notebook from the repository root, or run:

```bash
python -m examples.algorithms.classical_simulation
```

The example prints the original two-bit addition truth table, and checks
`a=5,b=7 -> a=5,b=12` through the real eight-bit ripple-carry graph. The notebook
also demonstrates ordinary `fq.Circuit(...).append(op)` with CNOT. These are
single-device numerical and local execution workflows, not scalability claims.

## Complete cell and asset disposition

Both notebooks contain 25 cells: 12 code and 13 Markdown. Positions below are
zero-based. All original code cells execute with numerical statements unchanged;
only optional `ClassicalSimGraphDrawer.get_svg` rendering is intercepted at the
test boundary. All native cells execute separately with imports of Qualtran,
Cirq and Quimb forbidden. The reference file remains byte-identical.

| Cells | Original responsibility | Native disposition |
| --- | --- | --- |
| 0,2 | Title and classical-call interface | Explicit named value callbacks, signature order and stored Core programs; Apache-2.0 attribution and locked source link. |
| 1,3 | One CNOT result and all four inputs | Shared `cnot_classical_values`, exact tuples and rows. Native cell3 additionally appends the real CNOT body and checks its state. |
| 4,5,7 | Consistency discussion | Direct numerical reference versus actual decomposition; basis samples are not claimed to prove quantum equivalence. |
| 6,8 | Eight-bit Add and ten seed52 pairs | Direct `register_add_classical_values` versus real `addition_component_program` through `ComponentClassicalState`. Split/join, CNOT, AND and checked erasure use existing numerical owners; no whole-word substitution. |
| 9,10 | Exhaustive consistency helper | All16 pairs from two two-bit domains compare direct action and actual graph. No foreign testing facade is copied. |
| 11,12 | Add truth table and format | Existing domain/table/format owners preserve exact rows, ordering and separator. |
| 13,14,15 | Action authoring and CNOTExample | Native imports and an explicit `(ctrl+target)%2` action replace foreign inheritance, attrs and wildcard typing. |
| 16,17 | Custom CNOT table | Exact original table text and all four results via existing signature/domain/table owners. |
| 18,19,20 | Classical-value graph drawing | Actual per-call assignments shown as a Matplotlib call/component/output table. This is value-flow inspection, not a Graphviz SVG or layout replica. Binding names are displayed as Word/bit binding; raw trace retains native names. |
| 21,22 | Big-endian bit encoding | Shared `integer_to_bits(254,8)` returns exactly seven ones followed by zero; MSB-first order retained. |
| 23,24 | Fixed-point integer representation and DoubleFxp | Exact rational scaling of0.25 to stored4, original doubling modulo16 to8, shared validation and scale decode to0.5. General float quantization is not introduced. Overflow discards information, so this action is not an in-place quantum permutation. |

All13 original Markdown cells have native teaching text referencing actual
behavior. Source Python3 kernel metadata is retained in normalized form; its
historical Python3.11.8 version is not copied as a runtime requirement. Native
IDs are unique and stable. Cell metadata has no execution dependency; counts
are null, outputs empty, and neither asset has attachments. There are no saved
source figures to copy. One native figure is generated from current assignments,
checked as a PNG and visually reviewed.

## Ownership and interfaces

Compiler remains authoritative for the factorized Add graph, typed bindings and
measured physical lowering. Simulation owns value actions, codecs, validation,
assignments and truth-table formatting. The example composes these owners: it
recognizes only its explicit Add leaf names and canonical split/join definitions,
refusing unknown recipes. It adds no production adapter, registry or graph.

Arrays traverse row-major; bit codecs and word bindings are MSB-first. Numerical
carry outputs are checked before erasure. Physical allocation, feedback and
release stay with the existing Add emitter. The trace owns no physical work.

The source fixed-point conversion delegates to a richer quantization dependency.
This notebook preserves its single exactly representable input with explicit
rational scaling; it is not a general `QFxp.to_fixed_width_int` replacement.
No SVG API or foreign Soquet identity is promised by the value table. These are
explicit asset adaptations, not entire datatype/drawing-package equivalence.

## Numerical and physical evidence

Tests execute all12 original and all12 native code cells, comparing original
CNOT/Add results, both table strings, all ten seeded pairs, all16 small-domain
inputs, bit encoding and fixed-point integer/real results. Widths1,2,3,8 also
compare every source decomposition call's complete output values, including
arrays and erased carries. The rendered table contains every call and has
visible image content; its text and layout are visually inspected.

The two-bit physical Add consumer compares emitted gates, operands and parameters
with unchanged source decomposition. Its full16-input complex superposition is
checked in both unnormalized measurement branches, with carry and unused
spectator zero. Four ordinary-append trajectories match the normalized target.
Existing Add tests additionally retain feedback metadata, larger-width primitive
comparisons and all small forward/inverse histories.

| Account | Result |
| --- | --- |
| Two-bit actual stream | H3,T2,Tdg2,CX9,S1,measurement1,conditional CZ1,reset1 |
| Independent upstream QEC model | AND1,Clifford4,measurement1; not primitive syntax counts |
| Complex-state test storage | Four data bits, one clean carry, one unused clean spectator; one classical measurement slot |
| Numerical state/table/codec/plotting | Zero emitted quantum primitives; arrays and assignments, no physical workspace |

Fixed-point doubling is not synthesized as an in-place unitary because it
loses information. No misleading quantum resource estimate is attached to it.

## Acceptance boundary

This disposes the complete concrete Q-0881 notebook asset. General foreign
interfaces, Core admission gaps, empty-array differences, input-sentinel identity
and lazy-decomposition diagnostics remain under Q-0882/Q-0883 and their
[test matrix](CLASSICAL_SIM_TEST_MATRIX.md). Notebook completion does not close
those implementation/test parents or start G3.

Local acceptance: six new cases;1,622 related tests in34.75s (24 numerical and
other warnings);84 CPU/TN checks in5.41s (two optional dependency/deprecation
warnings);28 unchanged source-only tests in1.22s (one notebook deselected, two
collection/marker warnings). Strict mypy covers both Python files; Black/Ruff,
scope, architecture, IR, API and three example scripts pass. Helper coverage88%
(94 statements,20 branches); unsupported-recipe/error and standalone-only paths
are not all covered by the instrumented test run. Default-suite timeout evidence
is retained without repeating that run; no full-suite green claim is made.

Implementation acceptance: `ce8bc24a`. Q-0881 closes after the complete asset
audit; Q-0882/Q-0883 retain the general implementation boundaries above.
