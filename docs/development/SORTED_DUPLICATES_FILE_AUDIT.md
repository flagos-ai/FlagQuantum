# Sorted duplicates complete source and test audit

Q-0201 and Q-0202 track the frozen implementation and original test file.
Their exact blobs are asserted by `test_sorted_duplicates_file.py`.

| Full source responsibility | Native owner and reused evidence |
| --- | --- |
| length/dtype/control fields and signature | `sorted_duplicates_signature` preserves signed/unsigned shaped xs, flag and final optional ctrl, including symbolic metadata |
| half-comparison dependency and graph | existing `sorted_duplicates_ir` follows adjacent half-LT, MCX, X/CNOT shell and reverse erasure, with exact gate/operand/feedback and alias-map comparisons |
| self adjoint | source returns itself, native append twice restores the valid sorted logical subspace on the preserving-order route |
| classical callback | Simulation `sorted_duplicates_classical_values`, with original shallow dictionary, ndarray aliases, sorting assertion, augmented XOR and ignored ctrl |
| specialized control | existing emitter positive-control input reuses shared MCX and controlled shell, both actual control inputs covered |
| three factories and DocSpec | concrete 4 by 3, symbolic word width and symbolic list/word widths all retain signatures, DocSpec contains only symbolic-width and concrete examples |
| original tests | original factory/count/classical tests execute unchanged, native suites independently verify actual gates, quantum amplitudes, scratch and raw measurement histories |

The untouched original test file reports 22 passed, 3 skipped and 1 xfailed,
with two existing unknown-slow-marker warnings. The fixture's skips/xfail
remain visible. Its classical composite tests pass because child callbacks
retain their logical contracts. They do not certify the original primitive
alias/erasure channel, whose counterexamples remain in the native gate suite.

The accepted 4 by 3 signed controlled circuit emits 48 T/TDG, 206 CX and 12
measurements on 32 physical wires. Its source model independently predicts
48 T equivalents, 12 measurements and 32 peak logical wires. Larger symbolic
root metadata is representable, while generic symbolic executable admission
and the original symbolic signed SignExtend width-query failure remain open.

Source aliases reorder [0,1] to [1,0]. Three-word [0,0,1] erasure can violate
its temporary-And target precondition. The uncontrolled singleton source
composite fails when X receives an empty controls port. Its native preserving
route is identity. The controlled raw callback ignores ctrl, unlike the quantum
control. Both interfaces preserve their documented distinct behavior. These
source/admission defects stay open under Q-0201 and Q-0202. No reference code
was repaired or replaced to claim equivalence.

Batch verification passed 718 concentrated native/reference cases with no skips or warnings.
Changed Python passed strict mypy, Black, Ruff, Compiler scope, architecture,
multilevel IR and protected API checks. Supplemental compare/exchange and
shared-control results are recorded in `g2-lists-file-control-dependencies.log`.


## Full declared model and lifetime closure

`sorting/sorted_duplicates_resource_model.py` owns original concrete component
costs, symbolic formulas and independent gate queries. Its four direct calls are
half-LT, half-LT adjoint, MCX and the X/CNOT shell. Comparison owns the half model,
which reuses the original widened out-of-place adder and three complements.
Signed extension contributes its actual four one-bit fanout calls. Measurement
categories come from the shared exact adder-adjoint model.

All three original factories and both controlled metadata variants match every
QEC category. Bound list lengths retain each comparison's live difference/flag
history, post-call snapshots, and the MCX peak. Symbolic lengths use the original
root/child fallback. Symbols retain the original absence of positivity
assumptions, including the zero baseline in the controlled singleton schedule.

The original uncontrolled singleton peak query fails at its empty control port.
Signed symbolic comparisons fail in SignExtend. Native peak APIs report those
specific failures, while independent original QEC queries remain available and
match source. The controlled singleton avoids the unused half-comparison query
and remains supported even for signed symbolic words.

Concrete model queries cover signed/unsigned widths one through three, lengths
two through four and controlled/uncontrolled modes. The actual four-word,
three-bit signed controlled source model predicts 12 logical And, 218 Clifford,
12 measurements and 32 peak wires. The independent native emitter retains its
accepted 48 T/TDG, 206 CX, 12 measurements and 32 physical wires. Its clean ordered
route and diagnostic alias route retain their separately recorded contracts.
The source defects and protected generic execution gaps remain open.

Batch acceptance passed 1115 concentrated native/reference cases with no skips.
The one existing IPython temporary-directory warning remains recorded. Changed
Python passed strict mypy, Black, Ruff, Compiler scope, architecture, multilevel
IR and the protected API baseline. Frozen source blobs remain unchanged.

## Stored graph interface and ownership

`sorting/sorted_duplicates_component.py` owns the concrete HasDuplicates
component hierarchy. `sorted_duplicates_program(length, dtype, controlled=False)`
retains the shaped MSB-first `xs` register, dirty XOR `flag` and optional final
`ctrl` port. Its child calls use the shared half-LT, exact half-LT inverse and
MCX owners. Each half call owns its difference/flag until reverse consumption.
The graph follows returned half-inverse aliases at every adjacent pair, exactly
as the frozen source. An uncontrolled singleton retains the source decomposition
failure. The controlled singleton has no comparison history and emits two CX.

This literal graph is separate from the existing physical-order-preserving
`sorted_duplicates_ir` route. Lists of three or more may reach the known original
invalid And-erasure precondition. Source-graph conformance must retain that
failure witness. Clean-channel claims apply only where the original computation
satisfies its erasure preconditions. Existing mathematical duplicate-detection
examples continue using their already accepted preserving route.

| Complete frozen definition | Native disposition |
|---|---|
| `HasDuplicates`, `HasDuplicates.signature` | Existing `sorted_duplicates_signature` retains shaped xs, dirty flag, optional final ctrl and symbolic declaration metadata. |
| `HasDuplicates._le_bloq`, `HasDuplicates.build_composite_bloq` | `sorted_duplicates_program` stores every original half-LT, MCX, shell and reverse-half call with exact named operands and source alias propagation. |
| `HasDuplicates.build_call_graph` | Existing `sorted_duplicates_resource_model` retains all direct concrete/symbolic child counts and lifetime models. |
| `HasDuplicates.on_classical_vals` | Existing Simulation `classical_sorted_duplicates` retains sorted-axis assertions, array aliases and the original controlled callback's ignored ctrl. |
| `HasDuplicates.adjoint` | Original adjoint method returns the same instance; its stored call body and existing literal-source versus preserving physical routes retain their separate contracts. |
| `HasDuplicates.get_ctrl_system` | Existing specialized one-bit control protocol and actual controlled signature retain their accepted boundary. Concrete controlled root now has a stored body. Generic outer control admission remains separately tracked. |
| `_has_duplicates`, `_has_duplicates_symb`, `_has_duplicates_symb_len` | Existing three-factory declaration/notebook audit is reused. The actual concrete factory additionally uses all five native graph checks. Symbolic executable admission remains open. |

The new stored-graph evidence covers whole root connection traces and complete
source primitive dependencies/operands/feedback through four three-bit words.
Every applicable two-word raw measurement history preserves full complex
amplitudes, dirty flag/control, spectator coherence and clean work. Three-word
one-bit channels are compared branch by branch with original source primitives,
without claiming their invalid erasure implements the ideal sorted-list oracle.
The existing two-bit `[0,0,1]` original invalid-And witness is reproduced by the
new graph. Original uncontrolled singleton failure and controlled singleton's
two-CX identity are preserved. Existing mathematical preserving-route, raw
callback, symbolic model, source-test and notebook evidence is reused.

Stored graph resource receipts retain the previously accepted four-word,
three-bit signed controlled counts of 206 CX, 48 T/TDG and 12 measurements.
The independent model has 12 And, 218 Clifford, 12 measurements and peak 32
quantum bits. The new Program uses twelve fresh SSA measurement slots. Logical
source Clifford categories and expanded primitive CX remain separate.
Ownership returns every temporary allocation, while physical cleanup assertions
remain restricted to source-valid erasure paths.
