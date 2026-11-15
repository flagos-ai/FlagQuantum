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
