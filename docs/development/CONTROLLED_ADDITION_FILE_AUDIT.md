# Controlled addition whole-file audit

The complete frozen Qualtran v0.7.0 `arithmetic/controlled_addition.py` at
`bcb17bb25360ed68ce3bc084645db3e929799389` defines CAdd, its validators,
classical callback, symbols, composite body, call graph and specialized control
system, two factories and one documentation specification. Source and original
test Git blobs are checked against the pinned ledger.

## Authoritative owners and direct dependencies

| Complete source definition | Native responsibility and evidence |
| --- | --- |
| a/b dtype, default b and activation validation | Native typed signatures and emitter validation retain ctrl/a/b order, concrete widths and both integer-control polarities. Equal and unequal integer encodings use existing metadata owners. Native execution rejects bool and float wire/control values explicitly. |
| Classical callback | Simulation classical_controlled_addition retains inactive object identity, active width checks, NumPy arithmetic and signed modular interpretation. Its full existing source comparisons are reused. |
| Composite controlled load/Add/erase | Compiler controlled_addition owns explicit loaded words, carries and classical slots. Shared And compute/erase, dtype Cast bookkeeping and the authoritative Add child retain their owners. |
| Generic exact adjoint | Existing controlled_add_adjoint_ir emits loaded compute, original Add adjoint and loaded erase. Actual source captures and source traversal are compared in each direction. |
| Call graph and peak width | Native ArithmeticCall and resource owners preserve original And(m), Add(m,n), And-adjoint(m) edges, category costs and concrete versus symbolic width dispatch. Existing replaceable-child consumers are reused. |
| Specialized outer control | The source adds one conjunction/erase around a positive-controlled CAdd. Existing native predicate resources represent this model. General physical/static foreign controlled-component admission remains separate open work. |
| Small three-bit factory | Both actual captured directions preserve every gate, operand and feedback dependency, reversed physical word binding and CircuitIR JSON through ordinary append. Source model peak is twelve and the stream has twenty T/TDG and five measurements/resets. |
| Large thousand-bit factory | The original declares 1999 Toffoli equivalents. Native emission has 7996 T/TDG and 1999 measurements/resets, with composed peak 4000. Dense-state execution at this width is not claimed. The original peak query fails through the recursive Add decomposition, as recorded below. |
| Presentation | Existing addition drawer callback now accepts controlled_add and preserves ctrl/a/a+b port labels and short name. The original full drawing entry point fails because its callback signature predates the drawing API. Direct original glyph calls are compared independently and the native complete score renders normally. |

All physical words are big-endian. Loaded work has m bits and carries have n-1
bits, all initially and finally zero, with one distinct classical storage slot
per measured work bit. Named capture assigns slots by measurement encounter
order. Tests explicitly bind this ordering when comparing the direct native
emitter. They do not conflate measurement order with work-wire order.

The new small capture scenarios cover unsigned, signed and Montgomery words,
both activation polarities and both source directions. An independent source
tensor checks every complex amplitude with a coherent spectator. Every first
measurement history and every compute/inverse history pair is checked with its
unnormalized amplitude and zero work. Larger streams, unequal signed witnesses,
all source width/category queries, exhaustive small basis actions and shared
child replacement reuse the existing focused suites.

## Original test file disposition

The original file has eleven test functions. Factory creation and notebook
execution, decomposition/count equivalence over all ten width pairs through
four bits, exact 2n-1 Toffoli counts, exhaustive and seeded unsigned/Montgomery
classical checks, exhaustive and seeded signed checks, symbolic cost,
t-complexity consistency and an outer-controlled ten-bit declaration each
retain their source responsibility. Native emitter/classical/resource/adjoint
suites cover these computational consumers. The notebook is a separate complete
teaching asset. Original test modes and native execution are reported separately.

## Source defects and remaining boundaries

Q-0186.02 remains open. The original narrow signed classical callback can return
-1 while its actual Add circuit zero-extends the same input and returns +1.
The native gates retain the actual source circuit. No mathematical sign
extension is inserted into that circuit.

The original CAdd.wire_symbol accepts a Soquet, while the current original
musical-score drawer calls it with Register and index. Full drawing therefore
raises TypeError before a source score can be produced. Native glyphs are
compared with the original method using its own declared Soquet-style input.
A corrected source drawer is not used as an unmodified reference.

QubitCount for the thousand-bit factory reaches Add's recursive decomposition
and raises RuntimeError wrapping another RuntimeError and RecursionError at the
normal Python recursion limit. The source Toffoli model still evaluates.
The native composed peak is recorded separately, without altering that limit or
claiming the failed source query returned a width.

Symbolic CAdd explicitly refuses decomposition while retaining 2n-1 CCZ model
cost. Native physical capture rejects symbolic ports. Protected static
converters and generic outer-controlled physical admission remain open.
Q-0186/Q-0187 retain these source/admission boundaries. No reference source is
modified. Logs are `ftqc-reference-env/g2-controlled-addition-file-*.log`.

Concentrated native regression passed 827 cases without skips. Four warnings
are paired original/native NumPy overflow, existing IPython fallback and an
existing noninteractive addition figure. Both changed Python files passed
strict typing, formatting and lint. Compiler scope, architecture, multilevel IR
and protected API checks passed. The native complete score was rendered and
visually inspected.

The untouched original test file passed 169 cases. The notebook test was
explicitly deselected for its separately tracked asset migration. Three warnings
retain the original unknown notebook/slow markers. The separate original
notebook execution boundary is documented in its teaching audit.
