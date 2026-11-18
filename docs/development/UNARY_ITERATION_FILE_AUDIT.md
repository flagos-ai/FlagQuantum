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

## Explicit root-fold execution and cost overrides

The native indexed consumer and the lower-level nested traversal now accept an
explicit zero-selection recipe. When the source hook folds the first-axis root,
the recipe replaces the entire traversal before any multi-control predicate or
indexed leaf is emitted. Without that recipe, the previously recorded default
failure remains. The zero-selection callback owns its direction and child
storage. The independent model takes one whole-signature child model, omits the
unused indexed child table, and verifies that the override covers the complete
parent signature. The emitter and cost still preserve default caller behavior.

A concrete custom source subclass overrides `decompose_zero_selection` with S
or a controlled S phase. Physical selector width is two, with bounds one and
four. Zero, one and two control bits are tested in both original directions.
The actual source directions are captured into native snapshots at the existing
Ecosystem boundary. The zero-selection recipe consumes those owned snapshots,
so every emitted instruction, operand and feedback matches the source stream.
The source controlled inverse uses its original measured And decomposition.
Both the source snapshot and the appended native circuit check every logical
operator column, every applicable measurement branch, exact complex phases,
JSON IR and clean reserved work. Unused parent predicate wires remain zero.
The base-class failure and missing override diagnostics remain covered.

| Root override | Captured primitives | Independent source model |
| --- | --- | --- |
| S, either direction | One S or S†, no work or measurement used | 1 Clifford, signature width |
| Controlled S forward | One controlled phase, no measurement used | 1 And, 2 Cliffords, 1 measurement, signature width + 1 |
| Controlled S inverse | 4 T/T†, 1 measurement, exact source feedback | 1 And, 2 Cliffords, 1 measurement, signature width + 1 |

The source model's expansion and captured primitive inventory are separate.
The callback does not introduce an alternative one-gate inverse. Capturing
literal zero-width ports, unresolved symbolic extents and further dynamic
allocation/borrowing hooks remain open under Q-0597 and the existing interop
entries. This section supersedes the earlier custom-zero-selection gap for the
verified positive-width root-fold override.


## Literal empty-selection source bindings

The named register emitter admits concrete zero-bit THRU ports through an explicit
snapshot capture option. Existing snapshot consumers retain their positive-width
admission rules. Named arrays preserve their empty shape and row-major layout.
At least one physical target is present in this acceptance batch. Scalar-only
carriers and dynamic borrowing remain separate interop scopes.

The native indexed emitter, nested traversal and independent model dispatch a
wholly empty selection signature directly to the explicit zero-selection recipe,
without querying the fold hook or allocating parent predicates. Child storage
belongs to the recipe. Mixed empty/nonempty axes remain rejected by the ordinary
indexed traversal. Negative and symbolic widths remain rejected. The default
source implementation still raises when no custom recipe is defined.

The same source subclass acceptance now includes a literal BQUInt(0, 1) selector,
zero through two control bits and both actual source directions. Exact source
primitive equality, every operator column, all measurement histories, complex
phase, clean child storage and JSON round trips are checked through
`Circuit.append(op)`. Independent whole-signature resource models match the
source call graph. The same checks cover an omitted selection register tuple.
Primitive inventories retain the S/controlled-S rows above,
with the zero selector removed from the signature width. This closes the
literal-zero-port gap recorded in the preceding section. Symbolic execution and
additional dynamic subclass hooks remain recorded under Q-0597.
