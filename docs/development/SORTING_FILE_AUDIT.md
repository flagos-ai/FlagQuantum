# Register sorting whole-file audit

Q-0216, Q-0217 and Q-0218 cover the sorting notebook, implementation and
tests at Qualtran v0.7.0 commit `bcb17bb25360ed68ce3bc084645db3e929799389`.
The three original Git blobs match the frozen inventory.

## Ownership and contracts

Compiler comparison owns compare/exchange and offset traversal. Compiler
sorting owns bitonic merge and recursive sort. All four concrete paths use
big-endian unsigned words and a reusable clean comparison pool. The
caller owns quantum work and feedback slots. Forward comparison history
is a RIGHT logical output, beginning at zero. It encodes the exchanges.
The exact inverse consumes the valid result and history to restore the
input and clear the history. The clean work is separate from the history.

| Original definitions | Native disposition |
|---|---|
| Comparator signature, composite and two factories | Preserve THRU a/b, RIGHT out and clean allocation followed by GreaterThan and exact CSwap. Existing gate, output and inverse tests remain authoritative. Concrete and symbolic-width factory graphs retain all direct child identities. |
| ParallelComparators signature, symbolic predicate, count, composite, graph and factory | Offset traversal stays in the shared block/step owner. All 81 original count/offset pairs now match original child input positions and history order, including empty blocks. Generic symbolic list lengths and offsets remain open. |
| BitonicMerge validation, signature, count, symbolic predicate, composite, graph and factory | Reuse the original cross comparisons, descending offset rounds and ordered history concatenation. The teaching view preserves LEFT xs/ys and RIGHT result/junk. Executable words bind the concatenated result to the original physical wires. The input halves must be ordered and their common length must be a power of two. |
| BitonicSort validation, signature, count, symbolic predicate, composite, graph and two factories | Reuse the original empty and pair bases and left/right/merge recursion. Preserve THRU xs and RIGHT ordered junk. Symbolic word width is available in the example declarations. Executable lowering retains concrete word/list admission. |
| Six factories and four documentation specifications | The catalogue covers every original full signature, comparison history shape, direct call edge and leaf multiplicity. Native graph names and tables replace the presentation objects. |

## Original tests and native evidence

The fourteen source functions cover four factory autotester groups,
manual and symbolic comparator costing, direct comparison values,
81 offset decomposition cases, seeded merges and sorts, the manual
24-comparison count, the fixed list and exhaustive four-element input
permutations, and the notebook. The original non-notebook run reports
114 passed, 2 skipped, 2 xfailed and 1 notebook deselected. The original
notebook passes separately. One skip concerns the comparator's
decomposition-only counts. The other retains the original strict
QUInt-to-QAny edge into CSwap. The two count-mode xfails concern merge
and sort and retain the upstream fixture's policy.

New native comparisons cover the six complete factory declarations,
the original 81 offset cases, and the source's 45 seeded lists across
all nine length/mode combinations, including 16-element sorts and
two 16-element sorted halves. Each seeded test compares both the data
and the complete retained history with the original classical composite.
Existing exact primitive certificates verify workspace cleanup for
every branch in these cases. Previously accepted gate dependency,
measurement feedback, full complex amplitude, spectator, inverse and
exhaustive small-history checks remain in the existing comparison and
sorting suites. This batch adds no alternate emitter.

| Eight-word, three-bit sort | Count |
|---|---|
| Actual primitives | 2376 CX, 288 X, 984 T/TDG, 120 measurements |
| Actual binding | 24 retained history bits, 8 clean work wires, 5 reused feedback slots, 56 quantum wires |
| Independent source count | 24 comparators, 120 logical And and 72 CSwap, 120 measurements |
| Source cost conversion | 984 T equivalents with exact 7 T per CSwap, or 768 under the default 4 T conversion |

These accounts reuse accepted Q-0217.05 gate and model evidence. The
source's symbolic-width legacy comparator estimate is a separate
estimation path from concrete primitive execution.

## Teaching asset and closure

The frozen 36-cell notebook and native 38-cell notebook run in independent
kernels. The native kernel blocks reference imports. It retains all six
examples, four signature groups, four direct call graphs and the original
teaching topics. The final example executes merge and sort with ordinary
`circuit.append(op)` and verifies the complete complex state and inverse.

Q-0216 closes. Q-0217 and Q-0218 retain generic symbolic and foreign
admission, the original strict dtype and autotester count-mode gaps,
and unsupported concrete lengths as recorded above.

## Concrete native execution graphs

Compiler comparison owns directional BiQubitsMixer/SingleQubitCompare calls,
the equal-width LessThanEqual tree, GreaterThan and compare/exchange. Compiler
primitives owns the exact typed CSwap wrapper over the existing seven-T emitter.
Comparison also owns offset traversal. Sorting composes merge and recursive
sort using those owners. Inputs are concrete MSB-first unsigned words. Tree
ancillas are RIGHT outputs consumed by LEFT adjoints. Comparator history is a
logical RIGHT output retained through the whole network and consumed by its
inverse. Temporary comparison work starts and ends clean. Existing symbolic
factory declarations, numerical primitives, models and classical callbacks keep
their authoritative owners. Normal consumers continue to use circuit.append(op).


The complete concrete `sorting.py` hierarchy now has native stored bodies.
`comparison/tree_component.py` owns the directional mixer and final comparison
flags. `comparison/ordering_component.py` composes the equal-width comparison
tree and exact GreaterThan adjoint. `primitives/controlled_swap_component.py`
owns register Split/TwoBitCSwap/Join and preserves QBit for one-bit source ports.
`comparison/compare_exchange_component.py` retains history allocation, ordered
comparison and exchange, and inverse consumption. Offset, merge and recursive
sort have independent owners and reuse these children. Existing physical,
classical, declaration and resource owners remain shared dependencies.

The two normal append examples now bind the stored graph. Their clean-pool
order keeps caller-visible history contiguous while flag lifetimes are reused.
The inverse explicitly adopts valid history inputs before clearing and freeing
them. Empty offset networks and one-word sorts are identity. The comparison
tree entry is equal-width only, matching sorting's direct dependency. Generic
unequal-width comparison support is outside this concrete graph batch.

The native/source comparison reads every network's original named connections,
all 81 offset cases, and the complete physical streams through three-bit words,
with comparator streams through eight bits. It binds each untouched source
comparator to the native allocation lifetime and maps measurement events
bijectively to SSA slots. This retains source component order when global Cirq
flattening reschedules independent branches. Full complex source tensors cover
the directional tree flags, dirty comparison target, coherent swap, small
networks, entangled spectators and all applicable raw histories in each
independent direction. Clean work and restored history pass within 3e-12.
The original 45 seeded lists also execute the native hierarchy, compare all
history outputs with source, and restore every input through the inverse.

All four actual concrete source factories run the existing five native check
modes. Their primitive multisets match the previously accepted emitters. The
source's strict QUInt/QAny typing gap and count-mode policy remain recorded
above. Native check-mode acceptance does not close those source fixture gaps.

| Native stored eight-word, three-bit sort | Count |
|---|---|
| Actual primitives | 2376 CX, 288 X, 984 T/TDG, 504 H, 120 S, 120 conditional CZ, 120 measurements and 120 reset operations |
| Quantum ownership | 24 retained logical history bits, 8 reusable clean work wires, 56 total quantum wires |
| Classical storage | 120 fresh Program SSA measurement slots, compared with 5 reused slots in the explicit emitter |
| Independent source categories | 120 And, 72 CSwap, 1392 Clifford, 120 measurements, peak 56 logical wires |
| Independent T conversion | 984 at the exact seven-T CSwap convention, 768 under the source default four-T convention |

Measurement events, reusable classical storage, quantum history and clean work
are reported separately. Source analytical categories are retained alongside
actual primitive counts. Their category names do not imply primitive identity.

Batch receipts use `g2-sorting-native-*` in the external reference evidence
folder. Focused acceptance passes 179 cases and the concentrated regression passes
5354 cases without skips, retaining 12 existing warnings. Two local integration
cases include both changed append consumers. Strict mypy covers
13 changed Python files, and format, Compiler scope, architecture, multilevel
IR and protected API checks pass. The default suite retains its previously
recorded docstring-example coverage failure. Q-0217/Q-0218 remain open for
generic symbolic/foreign admission, original strict dtype/count-mode gaps and
unsupported concrete lengths. The frozen reference remains unchanged.

## Generic declaration batch interface

Compiler comparison owns ComparatorDeclaration and ParallelComparatorsDeclaration,
including unsigned MSB-first words, offset/count metadata and RIGHT history.
Compiler sorting owns BitonicMergeDeclaration and BitonicSortDeclaration, including
LEFT merge inputs, RIGHT results and the retained recursive history. Signatures,
source symbolic predicates and direct comparison counts are independent of finite
physical lowering. The catalogue consumes these owners. Physical program methods
reuse the existing directional stored graphs, including their exact adjoints.
Logical history remains owned by the caller until inverse consumption. Each
child owns and clears its temporary comparison work. Symbolic lengths or widths
remain declaration/model values and are rejected by finite physical conversion.

## Complete declaration and consumer acceptance, 2026-11-20

This acceptance supersedes the generic-declaration and foreign-consumer items in
the earlier batch sections. Q-0217 and Q-0218 close with the original source and
physical-domain limits retained below.

All 29 source definitions are accounted for across the four classes, signatures,
validation, symbolic predicates, comparison counts, executable bodies, direct
models and six factories. Comparison declarations and sorting declarations have
separate owners. Both replace the catalogue's duplicated metadata. Every source
finite count/offset pair from one through nine compares its complete signature
and exact direct model. Generic symbolic count, offset, width and combined cases
retain the original half-count upper bound. Generic merge/sort symbolic length
and width retain the exact source bit-length expressions. Finite conversion in
both directions delegates to the existing accepted stored hierarchies.

All six actual source factories round-trip exact original library bytes at depths
zero, one and two and reconstruct equal source objects. Their real serialized
arguments resolve through general native declarations. Finite factories return
the existing executable graph. Symbolic-width factories retain declarations and
reject finite conversion. Symbolic-length libraries preserve source round-trip
assumption loss and the existing native array-dimension rejection. This foreign
resolution restriction is separate from symbolic declaration/model support.

Eight SDK consumers exercise all four classes through one- and two-bit words,
using real source sided signatures and explicit input/output qubit placement.
They compare complete source isometries including an entangled spectator, every
raw measurement history and zero non-output workspace through circuit.append(op).
These cases contain one comparator, with respectively two and eight raw histories.
The previously accepted complete networks retain their broader source hierarchy,
primitive operands/feedback, all small branch amplitudes, exact inverse and
45 seeded-list evidence. The original reference and all nine execution/interop
owners recorded in the receipt are unchanged.

The original direct BitonicMerge.as_cirq_op wrapper fails before native import.
For half-length one, it supplies 4n+1 physical operands to a wrapper expecting
2n+1. Both widths retain this exact original ValueError witness. Its unchanged
CompositeBloq.to_cirq_circuit_and_quregs path succeeds. The merge SDK execution
check imports that original composite circuit with explicit LEFT/RIGHT mapping.
The accepted native merge graph already retains the source physical output
aliases and clean comparison lifetime.

| One-comparator SDK case | Actual imported leaves | Independent source model |
|---|---|---|
| One-bit compare, offset, merge or sort | 15 CX, 8 X, 4 T/TDG, 1 measurement, 1 matrix CSwap leaf | 1 And, 1 CSwap, 14 Clifford, 1 measurement, 8 default-T or 11 exact-seven-T equivalents |
| Two-bit compare, offset, merge or sort | 45 CX, 10 X, 12 T/TDG, 3 measurements, 2 matrix CSwap leaves | 3 And, 2 CSwap, 36 Clifford, 3 measurements, 20 default-T or 26 exact-seven-T equivalents |

The SDK matrices are atomic unitary leaves. Their displayed T counts exclude
an expansion of those matrices. Full primitive expansion remains the accepted
native typed path using seven-T CSwap. The earlier eight-word table and
separate g2-sorting-native-evidence.json retain those measured primitives and
source categories. SDK counts are recorded in
g2-sorting-foreign-sdk-resources.json. Supplied unused clean-pool capacity is
identified separately from logical width in that receipt.

Concentrated regression passes 574 cases and local integration passes 18 cases,
including the normal append consumers. The 19 foreign cases pass independently
on Cirq 1.7.0 and 1.6.1. The generic declaration acceptance contains 102 cases.
Strict typing covers all five changed Python files. Black, Ruff, Compiler scope,
architecture, multilevel IR and public API checks pass. Five frozen blobs and
Qualtran 0.7.0 with Kickmix 0.1 Simulator are verified. Existing complete source
114-pass/2-skip/2-xfail and notebook evidence is reused. The prior default-suite
docstring-coverage failure retains byte-identical reported modules and test.

Original strict QUInt-to-QAny typing and count-mode policy remain visible.
Non-power-of-two merge lengths have declarations and models but no supported
source/native physical decomposition. Native declaration validation raises
ValueError for invalid lengths where the original constructor uses AssertionError.
Unknown symbolic dimensions retain metadata and direct models while finite
execution rejects them. The original merge SDK wrapper arity failure and source
symbol-assumption loss remain explicit. Receipts use g2-sorting-foreign-*.

## Complete concrete recursive presentation — 2026-11-22

All four concrete catalog roots now retain complete recursively qualified source
calls and fresh drawings. The frozen source and native definitions have 100
exact ordered graphs, including 96 non-root source-success contexts, and 749
atomic/refusal contexts. Every source-success descendant has its source class
label and an exactly matching fresh musical-score payload. The six original
sorting catalog factories retain their previously accepted immediate graphs.

The existing Comparator, ParallelComparators, BitonicMerge and BitonicSort
Compiler constructors remain authoritative. A domain-owned child provider reads
finite port dimensions and the first named comparator operand indices, then
requires literal equality of the entire reachable native definition. Shared
GreaterThan presentation belongs to the comparison provider. No function-name
heuristic, replacement executable body or alternate physical recipe is used.

A two-word parallel block and a two-word sort have identical bodies. A
BitonicSort label for that body requires an exact recursive sort parent with
both child positions referring to it. Standalone ambiguous bodies retain their
unqualified presentation. Tests cover this boundary, multiple observed parallel
offsets and altered measured feedback in all four complete wrappers.

| Concrete catalog root | Actual emitted T-family | Actual measurements | Source default 4-T CSwap estimate | Source exact 7-T CSwap estimate |
| --- | --- | --- | --- | --- |
| Comparator, width 3 | 41 | 5 | 32 | 41 |
| ParallelComparators, 7 words, offset 2, width 3 | 123 | 15 | 96 | 123 |
| BitonicMerge, half length 4, width 7 | 1212 | 156 | 960 | 1212 |
| BitonicSort, 8 words, width 4 | 1344 | 168 | 1056 | 1344 |

Full primitive histograms and independent source cost objects are preserved in
`g2-sorting-child-resources-20261122.json`. Every execution owner is byte-identical
to published baseline cd26f66a8. Presentation contributes zero physical gates.
The concentrated execution suite reruns source operands/feedback, complete small
complex tensors, applicable raw measurement histories, spectator entanglement,
forward logical histories, inverse history erasure and zero temporary work.
Forward logical sorting histories remain outputs until the separately verified
inverse consumes them.

The whole Arithmetic intake covers all 74 original catalog factories, recording
308 exact contextual graphs, 1521 original atomic/refusal contexts, seven native
declaration traversal boundaries, two duplicate-detection graph gaps and 27 roots
without a native UI decomposition. These are contextual graph counts on baseline cd26f66a8 with this presentation
batch applied. The complete receipt is
`g2-arithmetic-recursive-intake-complete-20261122.json`.
The latter roots retain separately accepted executable/declaration owners where
available. Five of those 27 roots have successful original decompositions and 22 have
original atomic/refusal outcomes. This UI inventory preserves their separately
accepted executable/declaration owners where available.

The symbolic comparator's GreaterThan child and the symbolic four-word sort's
BitonicSort and BitonicMerge children still have native declaration-only bodies.
Their original source decompositions succeed. The recorded IndexError comes
from attempting to traverse those native empty blocks. This is a native
recursive-graph gap, separate from original decomposition refusals. Symbolic
root drawings retain their immediate acceptance and finite execution rejection.
They are excluded from the 100 concrete recursive graph claim. Remaining
permutation, duplicate detection, symbolic dependency coverage, nominal type
views and pending contract proposals stay in the shared G2 ledger.

Acceptance passes 694 concentrated catalog checks, 708 execution/reference
checks and 15 focused presentation checks. Strict typing covers four changed
Python files. Black, Ruff, Integration scope, architecture, multilevel IR,
public API and whitespace checks pass. The catalog retains three existing
selection exclusions and fourteen notebook escape warnings. Execution retains
six original serialization deprecation warnings. Export runs are serialized
within the available disk budget. Combined catalog topology stays at 3569 JSON,
234 documentation files and 3568 diagrams, with 3215 contextual children,
68 decomposed roots and 285 atomic interfaces. Qualified contextual descendants
increase from 204 to 234. Other families' constructor matches retain separate
fresh-source acceptance boundaries.

## Symbolic recursive dependency closure — 2026-11-22

The three declaration-only sorting contexts recorded above are closed. All six
source factories, including the symbolic comparator and symbolic four-word sort,
now retain complete source-success dependency graphs down to original atomic or
refusal boundaries. The fresh traversal has 112 exact ordered graphs, 106 exact
non-root source class labels and fresh musical-score payloads, and 765 original
atomic/refusal contexts. Symbolic GreaterThan retains its LessThanEqual then X
calls. The unresolved LessThanEqual and CSwap leaves retain declarations at the
original physical decomposition boundaries.

Compiler `sorting/network_graph.py` now owns finite-array wiring for parallel
comparison, merge and recursive sort. The existing physical component factories
supply their accepted executable comparator. Symbolic examples supply a typed
comparator with the source comparison/swap graph. Both consumers share every
pair index, intermediate array binding and logical history connection. The old
example-only recursive sorting wiring was removed. The native component factory
signatures, width-admission boundary and exact inverse path are retained.

Independent published-baseline replay compares all 162 complete Program objects
across three physical factories, multiple finite dimensions, widths 1/2/3 and
both directions. Every Program equals baseline 86e0cda8c. Supplying an independent
history-producing comparator changes all retained history outputs in all three
shared consumers, proving the dependency is replaceable through the actual
binding path. This replacement fixture has its own explicit behavior and makes
no sorting-correctness claim. Existing native/source sorting execution consumers
retain normal `circuit.append(op)` acceptance.

All four concrete catalog primitive histograms and independent source cost
objects equal the preceding published receipt exactly. Actual T-family and
measurement pairs remain 41/5, 123/15, 1212/156 and 1344/168. Default 4-T CSwap
estimates remain 32, 96, 960 and 1056, with exact 7-T CSwap estimates equal to the
emitted T-family counts. Unresolved symbolic widths have dependency metadata and
models, with finite physical execution rejection. No physical primitive count is
assigned to an unresolved-width stream.

Whole-Arithmetic traversal now records 318 exact graphs, 1535 original
atomic/refusal contexts, four remaining native declaration-only source-success
contexts and the two duplicate-detection graph gaps. The 27 omitted roots still
contain five successful source decompositions and 22 original refusals. Remaining
symbolic declarations belong to BitwiseNot under negate/subtract-from and two
PermutationCycle contexts. Those separate source-file groups and other UI gaps
remain open under Q-0120. Three sorting symbolic contexts are fully closed at the
source-success graph boundary. Unknown array lengths retain their existing
finite traversal restriction and original symbolic/source failure evidence.

Acceptance passes 699 concentrated catalog checks, 708 execution/reference
checks and 44 focused shared-graph/presentation checks. Strict typing covers all
11 changed Python files. Black, Ruff, Compiler scope, architecture, multilevel IR,
public API and whitespace checks pass. Five frozen source blobs are verified.
The catalog retains three existing selection exclusions and fourteen notebook
warnings. Execution retains six source serialization deprecation warnings.

The Arithmetic contextual dependency graph grows from 2832 to 2853 definitions,
adding 21 formerly absent symbolic dependencies. The combined file totals follow
from the preceding measured export plus this exact node delta, with the complete
catalog file-equality regression covering the new exports. They are 3590 JSON,
3589 diagrams, 234 documentation files and 3236 contextual children. Qualified
contextual descendants increase from 234 to 244. The inventory receipt records
this derivation separately from literal fresh drawing comparisons.
