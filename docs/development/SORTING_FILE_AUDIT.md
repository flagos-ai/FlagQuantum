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
