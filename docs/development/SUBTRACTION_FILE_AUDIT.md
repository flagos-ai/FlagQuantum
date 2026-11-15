# Register subtraction whole-file audit

This batch covers Q-0219, Q-0220 and Q-0221 at Qualtran v0.7.0 commit
`bcb17bb25360ed68ce3bc084645db3e929799389`. All three original Git blobs
match the frozen inventory.

## Ownership and contracts

Compiler owns big-endian register lowering, word binding, clean prefix and
carry ownership, and classical feedback. `subtract_ir` preserves a and writes
a-b into b. For m <= n it uses n-m clean prefix wires, with sign extension
through the shared MultiTargetCNOT tree when requested. It applies the source
complement/add/complement sequence. `subtract_from_ir` uses equal widths and
writes b-a into b by complementing b before and after the shared adder.
Both bindings preserve a and return all quantum work to zero. The first
direction is an involution. The other is inverted by shared addition.

Simulation owns value callbacks. Subtract retains arithmetic before integer
conversion, the centered signed modulo and the left dtype's interpretation.
SubtractFrom converts operands and width first and then calls shared integer
addition. Existing tests cover NumPy conversion and overflow, symbolic
literals and unresolved widths. The local execution path is
`circuit.append(op)` with caller-supplied wires and slots.

| Original definitions | Disposition |
|---|---|
| Subtract dtype defaults, two validators, equal dtype accessor, signature and three unsigned projections | Native signed/unsigned/Montgomery bit-pattern admission and neutral register metadata. Exact foreign constructor coercion and generic dtype validation remain separate. |
| Subtract classical values and wire labels | Existing Simulation callback. Notebook uses the shared native signature and graph renderers. Arbitrary foreign wire-symbol callbacks remain outside the example contract. |
| Subtract direct declarations and composite build | Original Add and repeated X identities, width delta, allocation/free, and two sign-extension calls are retained in the signed example graph. Native compiled gate dependencies, operands, measurement and feedback already match the frozen concrete source. |
| SubtractFrom validator, signature, values, labels, declarations and composite build | Existing equal-width compiler and Simulation callback. Native signed example declarations retain typed BitwiseNot and Add. |
| Five Subtract factories, three SubtractFrom factories and two documentation specifications | All eight signatures and one-level graph edges/leaf counts match the immutable originals. The equal-width symbolic composite also matches both casts in each direction, all four arithmetic/complement calls, the eight-step insertion order, the active a/b inputs and final output bindings. This is an example resource and dependency view. Symbolic gate lowering and generic symbolic unequal widths remain open. |

## Original test disposition

All sixteen test functions and the signed conversion helper were reviewed.
Seven factory autotesters cover construction, decomposition, counts,
serialization and dtype checks. The unsigned/signed tensor matrix tests,
validation and explicit consistent-counts test, width 1–9 T model, value
comparisons, signed overflow, SubtractFrom decomposition and all small
classical cases ran against the original file. It reported 77 passed,
4 skipped and 4 xfailed. The skipped strict dtype checks expose source
QUInt-to-QAny edges following Cast or Join. The four Subtract count-mode
xfails are marked by the upstream fixture. The separate explicit consistent
counts test passes.

Native comparison reuses 39 concrete source gate-stream cases across unsigned,
signed and Montgomery words. Existing tests check per-wire dependency order
and the full primitive multiset, complex states with an entangled spectator,
all small measurement histories, zero work, and both restoration contracts.
No native emitter change is needed for this batch.

| Accepted scenario | Emitted counts | Model |
|---|---|---|
| Unsigned 3-to-8 Subtract | 81 CX, 28 T/TDG, 7 measurements, 12 work wires | 28 T equivalents and 7 measurements |
| Signed 3-to-8 Subtract | 99 CX, 28 T/TDG, 7 measurements, 12 work wires | Same T and measurement totals, separate sign-extension and Clifford accounting |
| Equal-width eight-bit SubtractFrom | 81 CX, 16 X, 28 T/TDG, 7 measurements, 7 carry wires | 28 T equivalents and 7 measurements |

Counts reuse prior evidence at `2c52ef80` and `f4193699`. Model Clifford
categories are separate from emitted CX and X totals.

## Teaching asset and open boundaries

The 24-cell source and 26-cell native notebook both execute in independent
kernels. The native kernel blocks reference imports. Native names and
signature tables retain all original topics, eight examples and both
displayed call graphs. The native notebook adds both append execution paths.

Q-0219 closes as a complete teaching asset. Q-0220 stays open for generic
foreign dtype/wire callbacks, controlled construction, and executable
symbolic decomposition. Q-0221 stays open for the original strict dtype
and generic count-mode boundaries above. The immutable reference is retained.
