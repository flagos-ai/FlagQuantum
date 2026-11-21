# Basis-state permutation whole-file audit

Q-0213, Q-0214 and Q-0215 cover the arithmetic permutation notebook,
implementation and tests at Qualtran v0.7.0 commit
`bcb17bb25360ed68ce3bc084645db3e929799389`. The original three Git blobs
match the frozen inventory. The new example catalogue and notebook reuse
the existing native emitters and resource owners.

## Ownership and contracts

Compiler owns big-endian register binding and ordered cycle emission.
The target x is a THRU bounded unsigned register. Each concrete cycle
compares its current entry into a marker and uses the marker to control
the XOR delta to the successor. Its final equality cleans the marker.
One marker, n-1 carry wires and n-1 reusable feedback slots are caller
owned. Every quantum work wire is zero at the end of each cycle. The
feedback slots retain the latest outcomes. Outside-cycle words are
unchanged. Overlapping cycles retain their declared order. The inverse
reverses cycle order and reverses each cycle.

The dense and partial cycle discovery owners preserve the locked order.
For the insertion-ordered partial map {0: 1, 2: 0}, the locked traversal
returns only (0, 1), losing the 2-to-0 edge. Reversing insertion order
produces (2, 0, 1). The original witness and broader partial-map checks
remain in `test_permutation_source_audit.py`. The checked native concrete
entry requires a positive register width, a fitting bound and a nonempty
sequence of legal cycles, with distinct values within each cycle.

| Original definitions | Native disposition |
|---|---|
| Two cycle converters, both cached signatures/default widths and symbolic detection | The example catalogue retains concrete tuples and the dimension of shape-only declarations, bounded x and the exact bit-length expression. Foreign constructors, general converter admission and foreign serialization remain open. |
| PermutationCycle executable body and graph | Existing `permutation_cycle_ir` retains marker allocation semantics, all equality and controlled XOR dependencies, singleton work and final cleanup. Graph evidence now compares the concrete constants, shape-unknown value, marker declarations, polarity and multiplicities for all three factories. |
| Permutation factories, build and graph | Dense/partial factories reuse ordered traversal. Example graphs retain ordered concrete cycles, the unknown-count worst-case length N and four indexed symbolic lengths. Generic from-cycle-lengths filtering and generic symbolic executable admission remain open. |
| Eight example factories and two documentation specifications | All original factory signatures, exact cycle data/shape, direct edges and leaf counts at depth one compare against the frozen source. Display uses native names, graph keys and signature tables. |

## Original tests and native evidence

All seven original functions were reviewed. Their modes cover eight
factory autotesters, six direct decomposition cases, concrete cycle and
multi-cycle tensor matrices and graph counts, sparse classical simulation,
and both symbolic count assertions. The unchanged source reports
37 passed, 8 skipped and 6 xfailed. Skips involve symbolic decomposition,
counts available through declarations only, dtype checks requiring
decomposition, and bounded-to-plain unsigned edges entering equality.
The six xfails concern source example serialization.

The native tests reuse the original exact concrete gate, operand,
measurement feedback and resource evidence recorded for Q-0214.01–.04 in the per-item ledger. Existing permutation
tests compare full complex states with an entangled spectator, exercise
each small measurement history, verify zero work and restore the full
initial state. Existing independent shape resource tests retain logical
cost and peak-width formulas. No executable gate recipe is changed by
this batch.

| Accepted scenario | Actual emitted subset | Independent model |
|---|---|---|
| Four-entry eight-bit cycle | 233 CX, 140 T/TDG, 35 measurements, 8 clean work wires, 7 reused slots | 35 logical And, 35 measurements |
| Ordered overlapping three-bit cycles | 101 CX, 56 T/TDG, 14 measurements, 3 clean work wires, 2 reused slots | 14 logical And, 14 measurements |

Primitive CX and source Clifford categories have separate accounting.

## Teaching asset and closure

The original 24-cell notebook and the native 26-cell notebook execute in
independent kernels. The native kernel denies reference imports. It
retains all eight examples, original teaching topics, both signature
groups and both displayed one-level call graphs. The final section runs
the existing coherent `circuit.append(op)` forward/inverse workflow.

Q-0213 closes. Q-0214 and Q-0215 retain the order-sensitive partial-map
witness, generic shape and foreign admission, strict source dtype checks
and original serialization boundaries. Symbolic declarations and costs
are available separately from concrete gate lowering.


## Native graph batch interface and acceptance

Compiler control owns literal MultiControlX through the existing shared predicate
and scalar CX owner. Compiler comparison owns EqualsAConstant through integer
Split/MultiControlX/Join. Compiler sorting owns ordered PermutationCycle and
Permutation graph composition. The target is a bounded unsigned MSB-first THRU
word. The cycle allocates and frees its clean marker. Equality owns and erases
its temporary predicate carries. Each cycle returns all work clean before the
next cycle. Concrete empty cycle sequences represent the source identity.
Existing source cycle discovery, symbolic declarations, resource models and
classical callbacks retain their respective owners and accepted evidence.


The three native owners preserve every comparable typed call boundary through
the accepted predicate, bookkeeping and primitive leaves. MultiControlX delegates
the source scalar X control directly to CX through the existing entry callback.
This avoids an additional stored wrapper. Equality retains the source integer
Split/MultiControlX/Join. The cycle retains Allocate, ordered equalities and
controlled XorK calls, the final equality and Free. Empty outer cycle sequences
retain the source identity. Original dense/partial conversion and symbolic
factory declarations reuse their existing owners and audit evidence.

The batch passes 77 focused checks. All three actual concrete source factories
use all five existing native example checks. Source root/child named connections,
primitive dependencies and feedback compare through eight bits. The small
source tensors include an entangled spectator and every raw measurement history,
for forward and inverse application independently. Workspace amplitudes are
zero within the same 3e-12 absolute tolerance as the full complex state. The
normal append example now binds these native graphs.

The three-bit overlapping-cycle example emits 101 CX, 52 X, 56 T/TDG and
14 measurements. Its independent source model is retained in the separate
seven-category resource receipt. Program allocates one SSA classical slot per
measurement lifetime, giving 14 slots in this example. The explicit primitive
interface reuses two caller-supplied slots. Each untouched source measurement
event is bound bijectively to its corresponding native event on that measured
wire, including the dependent feedback. Event count, SSA storage and reusable
primitive storage are recorded separately. Source primitive gates are unchanged.

The source whole-file 37 passes, 8 skips, 6 serialization xfails and complete
notebook evidence are reused. Q-0214/Q-0215 remain open for the insertion-order
partial-map lost-edge defect, generic shape/symbolic physical/foreign admission
and the original strict bounded-to-unsigned dtype and serialization boundaries.
Proposal 108 remains pending. Receipt prefix `g2-permutation-native-*` records
this batch, including the frozen source hashes and actual primitive/resource
model tables.


The final concentrated regression passes 4967 checks with no skips and twelve
existing warnings. Strict typing covers seven Python files. Black, Ruff,
compiler scope, architecture, IR and public API checks pass, as do six local
integration cases. The default suite reaches 434 passes and 51 skips before
the established docstring coverage failure. Its nine reported modules and the
failed test are byte-identical to base e2abfd8d4. Dependency policy reports
688 whole-repository findings and zero findings in this batch. Frozen source
blobs, Qualtran 0.7.0 and Kickmix 0.1 Simulator are verified separately.

## General declaration interface and dependency closure

Compiler sorting owns finite or symbolic permutation declarations independently
of physical execution. The declaration records the bound, width, ordered cycles
and complete shape-only cycle dimensions. Dense and partial factories delegate
to the existing cycle-discovery owner. Cycle-length construction preserves the
original symbolic-or-at-least-two filter. Signatures and direct model keys share
one owner with the example catalogue. Explicit physical conversion delegates to
the existing typed cycle/Permutation programs and retains their finite checks.
No shape metadata is interpreted as a list of known basis indices.

The foreign boundary preserves source bytes where original serialization
succeeds. Original Shaped/HasLength/object-array/ragged-array serialization
failures are verified before native deserialization and remain separate from
native declaration/model support. The source partial-map lost-edge example
remains unchanged.

## Complete source and consumer acceptance, 2026-11-20

This acceptance supersedes the open generic-construction and foreign-consumer
items recorded in the earlier batch sections. Q-0214 and Q-0215 close with
source failures and physical-domain limits retained explicitly below.

All 25 definitions in the frozen source are accounted for. The two converters,
cached signatures, default widths, symbolic detection and three factory methods
are implemented by the generic Compiler declaration and existing discovery
owners. Both executable bodies reuse the already accepted typed hierarchy.
Both direct models retain source child identities, insertion order and
multiplicities, including unknown-cycle-count worst-case length N. All eight
example factories and two documentation specifications retain the prior complete
catalogue and notebook evidence. Dense factory construction compares every
permutation through size five. Partial traversal preserves the original lost-edge
witness under both insertion orders. Length filtering preserves symbolic lengths,
duplicate multiplicity, empty outer identity and the original empty-cycle model
failure. Full shape tuples remain declaration metadata.

All eight actual source factories have a disposition at wire depths zero, one
and two. Concrete cycle and composed libraries round-trip exact original bytes
and reconstruct equal source objects. Symbolic-bound cycle bytes also round-trip,
while the source reconstruction loses positive/integer assumptions. Original
Shaped, HasLength, object-array and ragged-array serialization failures are
asserted before native decoding. Four finite/symbolic cycle/composed factory
resolutions consume the actual serialized arguments and original bounded-port
metadata. Symbolic declarations resolve without claiming finite gate binding.

Six SDK consumers cover two- and three-bit cycle, overlapping composition and
identity. Their real bounded signatures survive import. Ordered leaves, operands
and complete local matrices compare after the existing import normalization of
one/two negative controls. The test independently proves that basis change with
the original Cirq unitary before comparing streams. Complete complex source
tensors, an entangled spectator and zero quantum work are verified through normal
circuit.append(op). The SDK stream is coherent and has one empty measurement
history. Every raw measured-Bloq branch, exact adjoint, source call graph and
primitive dependency certificate reuses the accepted native batch evidence.

| SDK case | Original raw SDK leaves | Native leaves after negative-control normalization |
|---|---|---|
| Two-bit cycle | 72 | 15, including 3 atomic CCX |
| Two-bit overlapping composition | 142 | 28, including 6 atomic CCX |
| Three-bit cycle | 120 | 120, including 45 matrix leaves and 6 atomic CCX |
| Three-bit overlapping composition | 238 | 238, including 90 matrix leaves and 12 atomic CCX |
| Either-width identity | 0 | 0 |

Raw SDK leaf counts differ where control-basis normalization changes the selected
SDK decomposition. These counts contain atomic CCX and matrix gates. The measured
Bloq primitive counts and independent source model remain in the preceding
resource tables and g2-permutation-native-resources.json. The separate SDK receipt
is g2-permutation-foreign-sdk-resources.json.

Concentrated regression passes 372 cases and local integration passes 17 cases.
The 18 foreign cases pass independently on Cirq 1.7.0 and 1.6.1. The older SDK
emits 18 original NumPy array-shape deprecation warnings. Strict typing covers
four changed Python files. Black, Ruff, Compiler scope, architecture, multilevel
IR and public API checks pass. Frozen source hashes and Qualtran 0.7.0 plus
Kickmix 0.1 Simulator are verified. Existing execution, discovery, model and
foreign-boundary owners are byte-identical to the batch base. The prior default
suite's docstring-coverage failure is reused with every reported module and its
test verified unchanged against that recorded baseline.

The original partial-map lost edge, strict BQUInt-to-QUInt validation failure,
serialization exceptions, source round-trip assumption loss and original
37-pass/8-skip/6-xfail fixture disposition remain visible. Concrete physical
execution rejects unknown cycle values and symbolic widths. Accepted declaration
and model support is independent of this restriction. The frozen reference is
unchanged. Receipts use g2-permutation-foreign-*.

## Complete finite cycle graph closure — 2026-11-22

`permutation_graph.py` owns the finite marker/equality/controlled-XOR sequence.
The executable cycle owner retains domain validation and supplies its existing
child implementations. Symbolic catalog consumers supply the original typed
refusal declarations to the same graph. The word is MSB-first THRU storage.
The cycle owns one clean marker, equality owns its predicate workspace, and both
return their workspace clean under the accepted concrete execution contracts.
Unknown cycle values retain their original decomposition refusal.

All six source-success permutation factories now retain complete recursive
source graphs, including both concrete and symbolic-bound PermutationCycle roots.
The sparse symbolic-bound Permutation has real cycle bodies instead of empty
child declarations. Three frozen source blobs are verified unchanged. Thirty-two
complete concrete Programs equal published baseline b76ea2eb0 exactly. A supplied
XOR replacement reaches normal Program binding through the shared cycle graph.
Existing user entry remains circuit.append(op).

The original concrete and symbolic-bound cycle drawings raise ValueError with
"should return a `Text` object" despite successful decomposition. Root cycle
catalog diagrams retain the explicit native interface drawing independently of
the available dependency graph. Original unknown-cycle roots remain atomic.
PermutationCycle, equality, controlled-XOR and mixed-polarity control descendant
glyph qualification remains a separate catalog presentation gap. The nominal
unsigned-to-QAny equality Join view remains documented in the comparison audit.

| Concrete factory | Actual T-family | Actual measurements | Independent source model T-family / measurements |
| --- | --- | --- | --- |
| Permutation | 20 | 5 | 20 / 5 |
| Sparse permutation | 132 | 33 | 132 / 33 |
| PermutationCycle | 16 | 4 | 16 / 4 |

Full primitive histograms and analytical model objects are recorded separately in
`g2-permutation-cycle-resources-20261122.json`. Physical/refusal/source-resource
regression passes 197 cases with eight existing serialization warnings. It
covers exact primitive operands and feedback, complete complex states, applicable
measurement branches, workspace zero, source models and SDK boundaries. Accepted
execution evidence is reused alongside the fresh unchanged-Program comparison.

The first catalog regression stopped on ENOSPC. Its generated directory was
removed and two old regenerable mypy caches were cleared. Source trees and logs
are retained. The cleanup receipt is
`g2-permutation-cycle-cache-cleanup-20261122.json`.

Final catalog rerun passes 718 cases, with three existing deselections and 14
existing notebook warnings. Focused graph/provider checks pass 19 cases. Strict
typing covers six changed Python files. Black, Ruff, Compiler/integration scope,
architecture, IR, API baseline and whitespace checks pass. Full Arithmetic intake
now has 342 exact graphs, 1610 original atomic/refusal contexts and no successful
source/native declaration gaps. Two duplicate-detection graph gaps and three
successful omitted ControlledAddOrSubtract roots remain. Catalog descendant
glyph gaps listed above remain open independently of the permutation graph closure.

## Complete permutation descendant presentation — 2026-11-22

`examples/algorithms/permutation_child_decomposition.py` owns the catalog's
permutation-specific constructor selection. The owning cycle values select the
existing equality predicates and controlled-XOR deltas. Whole reachable
Program definitions must match literally before source class labels or glyphs
are used. The existing control provider retains ownership of mixed-polarity
conjunction and measured-adjoint glyphs. Matching does not infer constructors
from helper names or introduce an execution implementation.

All six source-success factories retain 114 exact recursive graphs and 297
original atomic/refusal contexts. Every one of the 108 non-root source-success
contexts is qualified with its original class label. Fresh source drawings match
for 102 contexts. The remaining six PermutationCycle descendants reproduce the
original Controlled/XorK title ValueError. Their complete native graphs use the
existing partition glyphs. The two cycle roots retain the previously documented
native interface diagrams and available dependency graphs. Reference code is
unchanged, with three source blobs reverified.

The complete descendant drawing gap is closed for this source family, with the
source drawing failures recorded as explicit exceptions. Root graph topology,
typed Programs, primitive emissions, numerical execution and resource models
are unchanged from accepted head dd5657078. Full Compiler/Core/execution owners,
permutation Program constructors and all nine execution acceptance test files
are verified unchanged. The preceding 197-pass execution receipt and separately
recorded actual primitive/model resources are reused. No fresh execution rerun
is claimed. Normal usage remains circuit.append(op).

Measurement-feedback mutation and unrelated cycle-constant metadata both reject
the enclosing cycle and equality class labels. These checks complement the
existing all-small-polarities control conformance and feedback-mutation tests.
Whole catalog totals remain 3737 JSON, 3736 diagrams, 234 documentation files,
3383 contextual children, 70 decomposed roots and 283 atomic interfaces. Qualified
contextual descendants increase from 252 to 336. Existing published export counts
are reused with an exact zero-node topology delta and complete catalog regression.

Final concentrated catalog regression passes 719 cases with three existing
deselections and 14 existing notebook warnings. Focused recursive/provider/control
regression passes 48 cases. Strict typing covers four changed Python files.
Black, Ruff, integration scope, architecture, IR, public API baseline and whitespace
checks pass. Receipts use `g2-permutation-glyph-*`. The next complete source group
is duplicate detection. Its fresh named-input mapping audit has 22 exact graphs,
88 original atomic/refusal contexts and zero graph gaps. Two half-comparison
class labels still require qualification. Raw predecessor-grouped mapping order
is retained as a witness, independently of named input semantics.
