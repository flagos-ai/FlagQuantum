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

## Native graph batch interface — 2026-11-20

The complete-file batch owns `Subtract` and `SubtractFrom` executable
component graphs in Compiler arithmetic. It reuses typed Add, BitwiseNot,
OnEach, Cast and word-lifetime owners. A shared bits fanout graph reuses the
existing `fanout_ir` source sequence and has no workspace or measurements.
Its control is QBit, targets are one opaque MSB-first word, and both remain live.

Subtract accepts concrete signed, unsigned or Montgomery a/b descriptors with
0 < width(a) <= width(b). It preserves a and writes a-b into b. Unequal widths
own an opaque prefix of width(b)-width(a), sign-extended only for signed a.
The original three OnEach complements, unsigned equal-width Add and all Cast,
Split/Join and prefix allocate/free calls remain explicit. Prefix work and Add
carries return clean. SubtractFrom uses equal-width typed inputs, preserves a
and writes b-a into b by BitwiseNot/Add/BitwiseNot. The user append entry remains
unchanged. Numerical conversion and execution stay outside Compiler.

The native root call sequence, named operands and all Cast/Split/Join calls
match the unmodified source. Add, OnEach and fanout implementation evidence is
reused at those dependency boundaries. Actual CNOT/T/measurement/reset and
conditional CZ streams are independently matched on unsigned, signed,
Montgomery and mixed descriptors through eight bits, including unequal widths.
The five concrete original subtraction factories (signed4, signed64, signed4
into signed16, and SubtractFrom signed4/signed64) use all five existing native
check modes. No replacement testing framework is introduced.

Full source complex tensors with an entangled spectator are checked at small
widths. Every raw erasure history and every restoration history pair agrees
without normalizing away branch amplitude or global phase. Prefix/carry work
returns zero. The a-b graph restores by a second a-b call. The b-a graph restores
through the accepted typed Add owner. The existing user append example now
binds both native component graphs.

Model and primitive counts remain separate. For signed2 into signed3 a-b,
measured emission is 23CX, 9X, 6H, 4T, 4Tdg, 2S, 2measurement, 2CZ and 2reset.
The independent source model is 2And, 22Clifford, 2measurement and peak8.
Three-bit b-a emits 21CX, 6X, 6H, 4T, 4Tdg, 2S, 2measurement, 2CZ and 2reset,
with source model 2And, 17Clifford, 2measurement and peak8. The shared fanout
emits 2n-1 CNOTs, while source QEC charges the entire fanout as one Clifford.
All seven model categories are checked against the accepted Add formula plus
source complement/fanout declarations. The existing whole-file source77pass,
4skip, 4xfail and notebook evidence is reused. Generic foreign callbacks,
controlled/symbolic admission and original strict dtype/count-mode boundaries
remain open in Q-0220/Q-0221.

Final local acceptance has 74 focused and 4,072 concentrated passes, with six
paired original/native NumPy warnings and two existing notebook/environment
warnings. Strict typing covers all five changed Python files. Black, Ruff,
team scope, architecture, IR, API checks, the append example and four related
integration checks pass. The default suite has 434 passes and 51 skips before
the established docstring coverage failure. Its nine reported modules and
failing test are byte-identical to base `bc082ff96`. Dependency policy reports
688 whole-repository findings and none in changed paths. Evidence receipts use
`g2-subtraction-native-*`. Q-0220/Q-0221 remain open for the explicit boundaries.

## Complete declarations and foreign closure — 2026-11-20

This section supersedes the earlier open-interface disposition for Q-0220 and
Q-0221. All 28 definitions in the frozen source have an owned implementation
or an explicit original-source failure. The previous ordinary graph, classical,
notebook and original-test receipts remain accepted evidence.

| Complete source responsibility | Owner and acceptance |
|---|---|
| Subtract default, validators, dtype accessor, signature and unsigned projections | `subtraction_declaration.py` retains signed/unsigned/Montgomery descriptors, default object identity, unequal-width validation, same-dtype accessor, Montgomery modulus and signed-to-unsigned projections. Original QBit/QAny/BQUInt refusal remains. |
| Both direct models and symbolic constructors | Generic declarations own ordered child models. Catalogue delegates to those owners. Equal and unequal symbolic widths retain source expressions and original finite-physical refusal. |
| Both classical callbacks and composite builders | Existing Simulation and native graph owners are reused. Scalar controlled calls derive the stored hierarchy and specialize Add as CAdd. Both control polarities are tested. |
| Both wire-symbol callbacks | `drawer/subtraction_symbols.py` supplies original formulas and sided glyphs through the existing typed callback. Complete register-score geometry matches the original. |
| All eight factories and both documentation specifications | Exact wire transport at depths 0/1/2. All finite factory resolutions bind identically to direct native programs. Symbolic leaves remain typed declarations. The actual symbolic composite resolves all eight stored calls and views. Existing notebook/documentation coverage is reused. |

The positive controlled graph preserves every reachable source call and named
operand down to already-accepted atomic dependencies. Negative control retains
the original X/positive-body/X wrapper. CAdd owns its predicate word and measured
carry cleanup. Unequal signed Subtract controls the original prefix fanout
through atomic Toffoli, while casts and lifetime bookkeeping remain unconditional.
A one-target fanout now calls CNOT directly, matching the source's absence of
Split/Join at that width. This fixes a structural difference found by recursive
call comparison. Wider fanout retains the original shared tree.

All four source integer/mixed families and both directions/control polarities
match original primitive counts, operands and measurement feedback through eight
bits. Complete complex source tensors with a coherent spectator match every raw
history at widths up to three, with clean prefix and carry storage. Tests use
ordinary `circuit.append(op)`. Twenty-four plain SDK cases retain the same streams,
and both positive-controlled SDK consumers match all 32 raw histories at width
three. Typed signatures are translated from the actual original descriptors.

| Concrete scenario | Measured emission | Independent source model |
|---|---|---|
| Positive signed 1-to-3 Subtract | 6 atomic CCX, 48 CX, 20 T/TDG, 15 H, 5 S, 5 measurement/CZ/reset each | 6 Toffoli, 5 And, 23 Clifford, 5 measurements, peak 12, default 44 T equivalents |
| Negative signed 2-to-3 Subtract | 2 X, 2 atomic CCX, 48 CX, 20 T/TDG, 15 H, 5 S, 5 measurement/CZ/reset each | 2 Toffoli, 5 And, 25 Clifford, 5 measurements, peak 12, default 28 T equivalents |
| Positive three-bit SubtractFrom | 45 CX, 20 T/TDG, 15 H, 5 S, 5 measurement/CZ/reset each | 5 And, 20 Clifford, 5 measurements, peak 12, default 20 T equivalents |

Atomic CCX emissions exclude a Clifford+T expansion. The source's default
four-T Toffoli model is separate from an exact seven-T expansion. The positive
SDK emission counts equal native counts at this common granularity. Negative
SDK export follows an original generic controlled measured-adjoint path whose
unitary/import recursively fails. Both original failures remain regression
witnesses. Native negative control is qualified against the unchanged source
Bloq decomposition, independently of that failing SDK export.

The symbolic composite's depth-zero source reconstruction loses its body and
is unequal to the original. Depths one and two reconstruct equally. Native
wire transport is exact at all three depths. Symbolic physical dimensions remain
unbound, and original strict QUInt-to-QAny/count-mode fixture failures remain
explicit. No patched reference is used to claim parity. Prior source 77 passed,
4 skipped and 4 xfailed and complete notebook evidence are reused.

Final batch acceptance has 810 concentrated and 19 integration passes. All 47
foreign/drawing cases pass on both Cirq 1.7.0 and 1.6.1. The older SDK emits 1,514
NumPy shape deprecation warnings. Strict typing covers nine changed Python
files. Black, Ruff, scope, architecture, IR and public API baseline checks pass.
The default-suite docstring failure baseline remains unchanged. Whole-repository
dependency findings and zero batch findings are recorded in the external
`g2-subtraction-foreign-evidence.json` receipt. Q-0220 and Q-0221 close with the
original failures and symbolic physical boundary documented here.

## Complete recursive word dependency closure — 2026-11-22

Symbolic SubtractFrom now reuses the existing BitwiseNot graph in both complement
positions. Its original Add declaration remains a source refusal. Symbolic
Subtract retains the accepted Cast/OnEach/Add/Cast order. Concrete Subtract and
SubtractFrom continue to reuse their existing executable owners verbatim.

All twelve source-success bitwise, negate and subtraction catalog roots retain
34 exact ordered recursive graphs, 22 qualified non-root class labels and 105
original atomic/refusal contexts. Twenty non-root source drawings match fresh
native payloads exactly. The two Add(width=64) descendants under Subtract and
SubtractFrom reproduce an IndexError in the frozen source musical-score builder.
They keep exact ordered graphs and labels, with the original drawing failures
recorded separately. Original source remains unchanged. Previous Add(width=100)
drawing failure evidence belongs to the earlier addition-file closure.

The two formerly empty symbolic BitwiseNot definitions are closed. Whole
Arithmetic traversal now records 320 exact graphs, 1537 original atomic/refusal
contexts and two remaining declaration-only source-success contexts under
PermutationCycle. The two duplicate-detection graph gaps and five source-success
omitted roots retain separate ownership. The existing symbolic XOR failure and
malformed CompositeBloq failure are still exercised directly.

| Concrete catalog root | Actual T-family | Actual measurements | Independent source T-family / measurements |
| --- | --- | --- | --- |
| Negate, 8-bit signed | 28 | 7 | 28 / 7 |
| Xor and BitwiseNot | 0 | 0 | 0 / 0 |
| Subtract, width 4 | 12 | 3 | 12 / 3 |
| Subtract, width 64 | 252 | 63 | 252 / 63 |
| Subtract, widths 4 and 16 | 60 | 15 | 60 / 15 |
| SubtractFrom, width 4 | 12 | 3 | 12 / 3 |
| SubtractFrom, width 64 | 252 | 63 | 252 / 63 |

Full emitted primitive histograms and independently evaluated source model
objects are stored in `g2-symbolic-word-resources-20261122.json`. Every Compiler,
Core and execution owner is byte-identical to published baseline 4cbfcd225.
Presentation adds zero physical operations. The focused execution regression
reruns original primitive operands and feedback, full small complex states,
spectator entanglement, applicable measurement histories, exact adjoints and
zero workspace. Unresolved widths keep metadata/models and physical rejection.

Final batch acceptance has 711 concentrated catalog passes, three existing
deselections and 14 existing notebook warnings. Execution/reference regression
has 718 passes. Focused recursion/provider regression has 108 passes. Strict
typing covers seven changed Python files. Black, Ruff, integration scope,
architecture, IR, API baseline and whitespace checks pass. Eight frozen source
blobs are verified. The two original 64-bit Add drawing failures remain explicit.
Combined export counts derive from the preceding published export plus the exact
two-node increase and complete catalog file-equality regression, recorded in
`g2-symbolic-word-inventory-20261122.json`.
