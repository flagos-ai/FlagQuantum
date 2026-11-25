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

## Actual foreign library and SDK consumers (2026-11-20)

The complete eleven-definition source inventory and its HalfLT, MCX, classical,
resource, direct graph and original notebook evidence are reused. The three
actual source factories now pass foreign-library checks in unsigned/signed and
controlled/uncontrolled forms. Concrete length/width and symbolic word-width
libraries retain exact deterministic bytes at depths zero, one and two. Explicit
native factories resolve all concrete roots into the existing stored duplicate
owner, with exact physical circuit and output aliases. Symbolic word widths
resolve declarations and retain physical-width refusal.

The original symbolic-length factory serializes only at depth zero. At larger
depths its unchanged serializer rejects HasLength(l-1). Approved Proposal 120 now permits native root resolution as a bodyless
symbolic array declaration. The original HasLength serialization failure and
unresolved physical execution remain explicit boundaries. Updated constructor
evidence is recorded in RESOLVER_LIST_AUDIT.md.

For two unsigned words of widths one, two and three, both control modes import
through the current typed SDK path. This SDK interface returns fixed physical
output ports, whereas the Bloq interface returns exchanged word aliases. SDK
primitive CCX is retained when the upstream SDK emits it. Independent untouched
SDK decomposition uses the existing workspace manager only to reserve live
sibling operands, retaining every original decomposition method. The comparison
matches complete temporary-wire histories and then checks every wire's ordered
gates, every operand, all measurement feedback metadata and total primitives.
Temporary names and independent global scheduling order may differ. The original
Bloq graph remains independently covered by its existing exact source walk.

Widths one and two additionally preserve all valid sorted input superpositions,
dirty flag, both control inputs and an entangled spectator through every raw
measurement history. Unnormalized branch amplitudes match the oracle and every
allocated work bit returns to zero. Source alias-defective and invalid-erasure
paths retain their previous witnesses. This batch adds no production owner.

Both signed SDK widths one and two fail in the original SignExtend Cirq wrapper
with a wrong-qubit-count ValueError. The untouched source alone reproduces the
same failure before native import, in both control modes. Native stored signed
execution retains its earlier source-graph, complex-branch and cleanup evidence.
The source was not modified to supply a successful SDK reference.

Resources are recorded per execution route. For two one-bit unsigned words,
SDK primitives contain 23 CX, 13 X, three H, two T, two T-dagger, one S,
one measurement, one feedback CZ and one reset. Controlled SDK has twelve X and
one additional atomic CCX, with the same four explicit T gates and one
measurement. Both SDK routes allocate six clean work bits. Independent source
models report one versus two And, one versus two measurements and peak seven
versus eight logical qubits. These source-model categories and the Bloq route's
existing measured costs are separate from SDK primitives and allocation. All
six SDK width/control cases contain zero matrix atoms. Full receipts are in
`g2-duplicates-foreign-resources.json` in the external reference directory.

Validation passed 799 concentrated cases and 17 local integration cases without
skips or warnings, plus 22 focused cases in both Cirq 1.7 and 1.6.1 lanes. The
unchanged complete source test file reports 22 passed, three source fixture
skips, one source xfail and two unknown-slow-marker warnings. Changed Python
passes strict mypy, Black and Ruff, with Compiler scope, architecture, multilevel
IR and protected API checks. Parent Q-0201/Q-0202 remain open for the explicitly
retained source failures and symbolic-length execution boundary.

## Complete recursive catalog dependency closure — 2026-11-22

The concrete and symbolic-width source-success factories now expose all 22
reachable ordered graphs and 88 original atomic/refusal contexts. Both
LinearDepthHalfLessThan descendants qualify through the existing half-comparison
owner, including forward and reverse port sides. Every decomposable child has a
whole-definition source class match and a fresh source-equal drawing. The
symbolic-length root retains its original decomposition refusal.

The earlier two apparent raw trace gaps were named-input dictionary insertion
order. The original iterator groups inputs by their producing predecessor.
Comparison now matches named inputs by name, retaining ordered calls and every
producer, port, index, output alias and measurement feedback value. The external
recursive receipt retains raw_named_mapping_order_equal=false for both roots.
The frozen source remains unchanged.

The combined complete has_duplicates.py and controlled_add_or_subtract.py batch
has 32 exact graphs, 113 atomic/refusal contexts and 27 fresh matching non-root
drawings. Its 381 execution/reference checks rerun complete complex amplitudes,
applicable measurement histories, workspace and primitive/model evidence.
The duplicate alias permutation and invalid three-word erasure precondition
remain witnessed. Clean-work acceptance applies to the already verified valid
two-word channels. Four-word resource counts alone do not certify a clean
mathematical duplicate channel.

The concrete four-word unsigned catalog stream has 44 T-family operations and
11 measurements. The independent frozen model reports 44 T equivalents and 11
measurements. Full primitive histograms and model categories are separate in
ftqc-reference-env/g2-arithmetic-tail-resources-20261122.json. Existing signed SDK
and symbolic execution boundaries above remain recorded. Parent Q-0201/Q-0202
remain open for those source and execution boundaries.

Batch verification passes 729 concentrated catalog checks with three existing
deselections and 14 original notebook warnings, 25 focused source/graph cases,
and 381 execution/reference cases with 100 existing protobuf bool warnings.
Strict mypy covers eight changed Python files. Black, Ruff, integration and
Compiler scope, architecture, multilevel IR, protected API and whitespace checks
pass. Four frozen source/test blobs remain unchanged. Combined catalog counts
are derived from the published export plus the exact 88-node Arithmetic delta,
covered by full catalog file-equality regression. They are 3825 JSON files,
3824 diagrams, 234 docs, 3471 contextual children, 73 decomposed roots,
280 atomic interfaces and 345 qualified descendants. Q-0120 remains open.


## Degenerate-domain dependency closure

The source controlled empty-list and singleton graphs contain MultiControlX(1)
followed by CNOT on the same control and dirty flag. The native typed graph
retains both CX operations. No half comparison is constructed for either graph.
Empty xs shape and zero-width signed/unsigned singleton words are retained
without quantum workspace, classical slots, measurement or reset. Uncontrolled
empty-list and singleton sources fail on the empty controls port and retain
bodyless constructor declarations.

For unsigned zero-width lists with comparisons, each half-LT still widens both
empty words to one clean bit, invokes the existing one-bit out-of-place adder,
retains its one-bit difference and flag, and consumes both through the original
inverse graph. All four half relations use this shared unsigned-empty admission.
Other linear comparison callers retain their existing positive-width validation.
Signed zero-width half comparison retains the original SignExtend IndexError
and explicit native physical refusal.

Source-derived operand streams are checked per-wire in dependency order, allowing
only interleaving of independent gates. Constructor binding preserves identical
native instruction streams. Coherent dirty flag/control and complex spectator
channels are checked for controlled lengths zero/one and unsigned zero-width
lengths two/three/four, with every applicable measurement history and all work
returned to zero. Ordinary Circuit and DynamicCircuit append paths execute.

Actual empty/singleton controlled emission is two CX and zero private work.
The frozen source QEC model for length zero has negative comparison multiplicity,
negative Clifford counts and, for positive word widths, negative And/measurement
counts. These original estimates remain available separately through
source_sorted_duplicates_gate_cost. The resource formula reports the actual
source graph peak of two logical qubits for controlled empty lists. Negative
model estimates are diagnostic data and are never substituted for measured
primitive inventories. Existing source three-word invalid And erasure and
deeper HasLength serialization failures retain their separate witnesses.

## Complete original test-file acceptance (2026-11-25)

Q-0202 now has a complete mapping for its three original functions. The full
unchanged original suite, including every slow case, reported 22 passes, three
skips and one xfail. The locked implementation and test blobs remain checked by
the existing native file fixture. Q-0201's implementation boundaries remain open.

| Original function | Complete native assertion mapping |
| --- | --- |
| `test_examples` | All three actual source factories and all fifteen raw protocol outcomes are observed before skip policy. Existing signature, model, library, native graph and SDK consumers retain their separate admission scopes. |
| `test_counts` | Both concrete and symbolic-width source count protocols pass; existing full direct child graph/model tests and actual factory resolutions retain operand and count identities. Symbolic-length counts remain unverified rather than a fabricated decomposition. |
| `test_classical_action` | Every original length 2/3/4 and QUInt(2), QInt(2), QUInt(3) domain, with both flag values, is compared through the actual original composite and native numerical callback. Enumerating each unique sorted tuple covers precisely the original sorted Cartesian inputs without repeating their permutations. |

The three factories produce eleven PASS, two NA, one UNVERIFIED and one FAIL
across fifteen modes. Symbolic-length serialization retains its source failure.
The original classical composite tests dispatch child callbacks; they are not
proof of coherent physical execution. Existing exact primitive/alias/feedback,
complete valid small complex-channel, measured-history and zero-work evidence
remains authoritative for its explicitly bounded scopes. Source three-word
invalid And erasure, swapped aliases, signed SDK failure, empty/singleton
behaviors and callback/control disagreement are unchanged under Q-0201.

The combined list assertion/component/model/constructor regression passes 673
cases; the independent Cirq 1.6 file/factory selection passes 55. Both complete
changed test files pass strict typing, Black and Ruff. Integration scope,
language, links, architecture, IR and protected API checks pass. Four frozen
source/test blobs and native Kickmix 0.1 Simulator are verified. Receipts use
`g2-list-source-closure-*20261125`; no source body was modified.
