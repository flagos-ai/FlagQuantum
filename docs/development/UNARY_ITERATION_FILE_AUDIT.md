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
