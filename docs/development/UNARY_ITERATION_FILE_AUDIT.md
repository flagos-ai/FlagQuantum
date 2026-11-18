# Unary iteration source-file audit

The reference is unchanged Qualtran v0.7.0 at commit
`bcb17bb25360ed68ce3bc084645db3e929799389`.

| File | Frozen Git blob | Ledger item |
| --- | --- | --- |
| `unary_iteration.ipynb` | `b0a18619c0b033d3870a24a4138c4b7400c70504` | Q-0596 |
| `unary_iteration_bloq.py` | `d1f689a99ecccaf9a44983d074875f2dce64ccb6` | Q-0597 |
| `unary_iteration_bloq_test.py` | `529640770dad7ee9ab564484c8cb725ad180e005` | Q-0598 |

## Standalone tutorial and binding

Open [the native notebook](../../examples/notebooks/unary_iteration.ipynb)
with `jupyter notebook examples/notebooks/unary_iteration.ipynb`.
Its 31 cells cover the original truth predicates, equivalence between
independent controls and the Boolean segment tree, the five-target
fully controlled example, the hand-worked four-target segment tree,
and the final bounded circuit. It verifies all original target selections,
prints compilations through the native drawer and emits the bounded
examples through `circuit.append(op)`.

The shared compiler traversal now admits a half-open address interval
`[lower, extent)`. The selector is big-endian. Children see the source
index and a positive predicate. The parent owns the clean predicate
chain and its classical feedback, and the child owns its own work.
An empty or inverted interval leaves the instruction list and child
unchanged. Entirely disjoint subtrees are pruned. Early folding applies
only to subtrees wholly contained in the interval. Reverse traversal
processes the retained leaves in reverse order.

## Original cases and resources

The comparison uses the locked single-axis fixtures at
`(width, targets, controls) = (3,5,1), (2,4,2), (1,2,3), (3,4,0)`
and nested extents `(2,3,2)` and `(2,2,2)`. It checks emitted gate
dependencies, operands, measurement feedback and independent source
resources. Each input covers all declared selectors, with all activation
patterns on controlled fixtures. Full complex outputs include clean
work. Small cases check each raw measurement history. Larger nested
fixtures check zero, one and alternating histories.

The original nested interval loop `[3,5) × [6,8)` calls the unchanged
source generator for the reference stream. The native circuit independently
applies the corresponding four toggles and matches all four measurement
histories. The empty and inverted intervals leave no operations.
An early-fold example visits source indices 3, 4 and 6 on `[3,7)`.

| Nonzero interval loop | Emitted primitives | Independent source model |
| --- | --- | --- |
| Counts | 8 T/T†, 18 CX, 2 measurements, 2 resets | 2 And, 10 Cliffords, 2 measurements |
| Width | 15 reserved wires | 11 peak live wires |

## Remaining source-file boundaries

Q-0596 and Q-0598 are accepted. All 15 frozen original cases passed,
including the independent notebook. Q-0597 remains open for symbolic
extent execution, arbitrary foreign Bloq subclasses and the general
custom decomposition and resource hooks. The current native interface
uses caller-owned controlled recipes and explicit work. Source pruning
is preserved even on padded bit patterns. The original interval contract
requires the input address to lie in the declared range.

## Prefix-aware custom subclass folding

The complete source base class, call-graph helpers, original test file and direct
nested traversal were reviewed. The native `append_nested_bounded_unary_tree`
now passes the retained parent indices to its early-fold callback. The existing
`indexed_component_ir` exposes that callback while retaining its controlled
recipe interface, flattened C-order leaf indices, caller-owned work and
feedback, and independent adjoint traversal. Default callers use their existing
path. A folded interval requires the same child recipe throughout the interval,
which is the original source subclass's responsibility.

A custom source subclass selects X or Z leaves on named `(row, column)` axes
with extents `(2, 4)`. Only row zero folds its column pairs. The actual source
base-class hook yields retained indices `0, 2, 4, 5, 6, 7`; the native forward
stream uses that order and the reverse traversal uses its reversal. Both streams
match source gate dependencies, operands and measurement feedback, including the
source's actual adjoint. The append entry, JSON roundtrip, every logical operator
column, all small measurement histories, full complex amplitudes and zero work
are checked with no control and one positive control.

The independent `indexed_component_cost` accepts the same pure fold callback.
Its source-style tree planning collects retained calls, split counts and live
predicate depths without emitting instructions or invoking the child emitter.
Skipped child models contribute no costs, and each retained call preserves its
own bystander width. It separately counts the outer multi-control predicate.
The default cost path is reused directly. Passing an always-false fold is also
checked against the complete existing call inventory, including uneven child
costs, padded extents, both directions and zero through three controls.

| Two-axis folded fixture | Emitted inventory | Source model |
| --- | --- | --- |
| No control | 16 T/T†, 4 measurements, 6 retained Pauli calls, 6 reserved wires | 4 And, 16 Cliffords, 4 measurements, 6 peak wires |
| One control | 20 T/T†, 5 measurements, 6 retained Pauli calls, 8 reserved wires | 5 And, 16 Cliffords, 5 measurements, 8 peak wires |

The original source calls `decompose_zero_selection` when the first-axis root
folds. Its base implementation raises `NotImplementedError`. The native emitter
and cost model require an explicit zero-selection recipe for that case and
refuse the ordinary tree route. The tests retain the source base failure.
Symbolic extent execution, custom zero-selection recipes and additional dynamic
subclass hooks remain open under Q-0597. Existing notebook, original test-suite,
interval pruning, nested-axis and reversed-traversal evidence above is reused.
