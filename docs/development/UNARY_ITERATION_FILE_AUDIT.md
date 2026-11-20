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

## Complete definition and private-child consumer audit, November 21

The complete implementation file and its directly dependent indexed execution,
resource and register-capture owners have the following dispositions. Existing
interval, root-fold, prefix-fold and empty-selection evidence above is reused.

| Source definition | Native responsibility and evidence |
| --- | --- |
| `_unary_iteration_segtree` | `append_bounded_unary_tree` preserves segment splitting, negative compute controls and positive measured release |
| `_unary_iteration_zero_control` | Existing input-predicate reuse and source flanking X operations |
| `_unary_iteration_single_control` | Existing positive predicate bounded traversal |
| `_unary_iteration_multi_controls` | Existing `indexed_component_ir` MultiAnd reduction and measurement-assisted cleanup |
| `unary_iteration` | Caller-owned workspace, half-open interval and leaf callback; native interval example and original tests are retained |
| `_unary_iteration_callgraph_segtree`, `_unary_iteration_callgraph` | Independent `indexed_component_cost` planning and prefix-depth live-width model; source graph comparisons remain separate from emitted instructions |
| `control_registers`, `selection_registers`, `target_registers` | Named control and bounded selector ports, row-major shaped targets and explicit child binding |
| `signature`, `extra_registers` | Existing named emitter and explicit native target ownership; a source subclass must include extra registers in its own signature |
| `nth_operation`, `nth_operation_callgraph` | Native controlled recipe and separate child resource model; private clean child lifetime exercised below |
| `decompose_zero_selection`, `_break_early` | Existing explicit root/empty recipe and prefix-aware folding, including original default failure |
| `decompose_from_registers` | Existing nested native emitter and real source subclass capture in both directions |
| `_circuit_diagram_info_` | Native register-score renderer owns presentation; original default diagram omits extra-register glyphs |
| `build_call_graph` | Existing independent child models with live parent predicates; original custom resource callback and child peak are exercised below |

The new consumer uses the unchanged source base class with a custom three-row
subclass. Its signature includes an optional enable bit, a two-bit bounded
selector, a target bit and a two-cell phase register. Each leaf allocates one
clean qubit, copies the positive predicate into it, applies CZ to the selected
phase cell and CX to the target, uncomputes the predicate copy and frees the
qubit. The source leaf separately declares three CNOT, one CZ and a four-qubit
peak. The native controlled recipe and child cost use the same existing
Compiler interfaces. No additional production owner is introduced.

Forward instructions match source per-wire gates, operands and feedback with
explicit parent and child storage. Every logical operator column, including
arbitrary phase-register values and the padded selector encoding, matches the
source's unnormalized complex amplitude in every measurement branch. All
parent and private child work ends at zero. The actual source adjoint receives
the same complete channel checks. Normal `Circuit.append(op)` and CircuitIR
roundtrips are exercised. Existing Ecosystem capture consumes the complete
subclass, converts both original directions to owned snapshots and executes the
same full-column/all-branch checks with its own classical slots.

### Retained private-allocation adjoint gap

The source generic adjoint replays Allocate/Free lifetimes and can recycle a
previous leaf's clean qubit as a later parent predicate. The fixed-child native
indexed recipe reserves one child wire throughout. Consequently their inverse
primitive names and counts match, and every complex branch and clean output
matches, while the raw physical work-wire operands differ. An exhaustive
permutation probe over the two- and three-wire work pools finds no single
whole-circuit wire renaming that aligns those inverse streams. Tests retain
this distinction explicitly. The frozen source and its allocation schedule
remain unchanged. Closing it requires lifetime-aware shared child/parent
allocation replay before raw inverse operand parity can be claimed.

The custom subclass also overrides `signature` to include its extra ports.
The original base signature contains only control, selection and target
registers. Its default circuit diagram likewise counts only those registers.
These are source subclass responsibilities and presentation limitations.
Unresolved symbolic physical execution, mixed empty axes and dirty borrowing
remain separate open obligations. Q-0597 remains open.

| Private-child fixture | Measured T-family / CX / CZ / measurement / clean work | Independent source AND / Clifford / measurement / peak |
| --- | --- | --- |
| No enable, either direction | 4 / 16 / 4 / 1 / 2 | 1 / 16 / 1 / 7 |
| One enable, either direction | 8 / 23 / 5 / 2 / 3 | 2 / 16 / 2 / 9 |

The shared source-oracle helper now uses the installed SDK's `CY` leaf when
available and the existing generic controlled-Y representation on Cirq 1.6.1,
which has no `CY` constant. Keeping the SDK-specific leaf is necessary because
Cirq 1.7's specialized decomposition differs from the generic representation.
The reference source itself is unchanged. The helper's manager is duck-typed,
matching the existing register snapshot manager, and explicit metadata types
and the existing DynamicCircuit variable preserve its strict typing.

The concentrated dependency lane passed 748 cases and local integration passed
17. The separate Cirq 1.6.1 lane passed 94. Strict typing of both changed test
modules, Black, Ruff, team scope, architecture, IR and protected API checks
passed. Frozen source hashes and unchanged production owners are recorded in
the external evidence receipt. The dependency-policy scan retains 666 existing
repository findings and zero changed-path findings for this batch.

## Typed shared-lifetime replay, November 21

The new `control/unary_component_calls.py` composes bounded iteration into the
existing `ComponentProgramBuilder`. Predicate computation and erasure use the
existing directional `and_component_program` definitions. Child code retains
its own ordinary Allocate/Free calls. The shared bounded walker now owns the
control flow for both this typed consumer and the existing physical IR
consumer. Default physical emission preserves its existing behavior.

The existing `adjoint_component_hierarchy` reverses the complete native call
graph with specialized And compute/erase definitions and exact unitary leaves.
The normal binder then allocates from the common live workspace pool. This
closes the preceding raw inverse allocation gap for the verified typed graph
path. Source and native CompositeBloq/Program routes agree on each primitive,
physical operand and per-wire order in both directions. Fresh native classical
slots are matched to the source measurement keys only after checking the
latest-write feedback lifetime. Source references use the unchanged original
graph exporter in both directions.

The direct forward Cirq generator reserves its full parent pool before visiting
children. The original typed graph can release and recycle those wires sooner.
These source entry points therefore have different allocation traces. Existing
physical IR tests continue comparing the direct generator, while the new typed
consumer compares the typed graph at the same granularity. The fixed-child
physical callback still preserves its caller's explicitly dedicated storage.

The complete private-child fixture from the preceding batch now executes as a
native Program. All logical columns and every measurement branch match the
original unnormalized complex amplitudes, including arbitrary phase-register
values and padded addresses. Parent and child work returns to zero, all
allocation/free events balance, outputs retain ownership and normal
`Circuit.append(op)` plus CircuitIR roundtrip work. Native Programs replace the
actual foreign snapshots behind the same `CircuitRegisterEmitter` consumer.
Construction, generic adjoint, binding and append also pass while imports of
Qualtran, Cirq and Kickmix are explicitly blocked.

Additional exact graph streams cover selector width/extent pairs `(1, 2)`,
`(3, 5)` and `(3, 8)`, each with zero or one enable bit and both directions.
The larger cases compare raw primitive operands, feedback, returned ownership
and independent source resources. Exhaustive quantum channels are certified on
the two-bit/three-row private-child fixture. Dedicated native cases additionally
verify nonzero interval bounds, two-entry folding, empty intervals and atomic
rejection of invalid bounds before consuming builder inputs.

Measured and modeled resource rows for the private-child fixture remain the
preceding 4/8 T-family, 16/23 CX, 1/2 measurement and 2/3 clean-work rows. The
independent source models retain 1/2 And, 16 Cliffords and logical peaks 7/9.
Q-0597 remains open for the remaining multi-axis dynamic-storage, mixed-empty,
unresolved-symbolic and dirty-borrow consumer obligations.

This typed-call batch passed 761 concentrated cases, 17 local integration cases
and 107 cases in the separate Cirq 1.6.1 lane. Strict typing of the three changed
Python modules, Black, Ruff, team scope, architecture, IR and protected API
checks passed. Three frozen source blobs and seven unchanged supporting owners
are verified. The dependency scan retains 666 repository findings and zero
changed-path findings. Model and measured resource rows remain separate above.

## Multi-axis and borrowed-child typed closure, November 21

The existing typed traversal now has a complete nested consumer matching the
original two-axis prefix-fold fixture. The outer row callback invokes a column
traversal, returns its successor predicate and retains the successor column
bindings for the next row. Row zero folds adjacent column pairs. All six
retained X/Z leaves pass through the existing native atom calls. Split/Join
retain the original bounded and opaque descriptors. The existing test adjoint
resolver matches every view with its opposite typed signature, including all
three Split/Join pairs. Production traversal, Program construction, exact
adjoint, binder and workspace owners required no further changes.

Both original CompositeBloq directions match native Program primitives,
physical operands, per-wire ordering and feedback lifetimes. Every logical
column and a dense complex superposition match in all measurement histories.
Clean work returns to zero, allocation/free ownership balances, outputs retain
their original wires and normal `Circuit.append(op)` plus JSON roundtrip pass.
The original direct-generator allocation profile remains a separate entry
point, as documented above.

The borrowed-child consumer uses the original unary base class with a typed
custom echo leaf. Its four-CNOT sequence restores the borrowed bit and toggles
the target by the positive predicate. The source fixture supplies an explicit
CompositeBloq with `Allocate(QBit(), dirty=True)` and the matching Free, as well
as its direct `qborrow` generator. This preserves the source dirty declaration
through the graph route. Native code uses the existing dirty Allocate/Free
functions and the existing distinct clean and borrowed binder pools. The test
oracle now accepts an optional independent borrowed pool. Existing callers
retain their original allocation behavior.

Both source graph directions match all primitive operands and feedback. All
logical columns, both dirty values and both spectator values match every
measurement branch. A dense complex state spans those joint columns and
certifies borrowed/spectator coherences against an independent controlled-X
permutation. Parent work returns to zero while the joint borrowed/spectator
state is restored. Each of the three retained children borrows and returns the
same physical dirty bit. The padded selector encoding follows the source's
pruned final leaf. The source's declared input domain remains the bounded
three-row interval.

| Typed consumer, either direction | Measured T-family / CX / CZ / measurement | Clean / borrowed work | Independent source AND / Clifford / measurement / peak |
| --- | --- | --- | --- |
| Folded axes, no enable | 16 / 31 / 7 / 4 | 2 / 0 | 4 / 16 / 4 / 6 |
| Folded axes, one enable | 20 / 38 / 8 / 5 | 3 / 0 | 5 / 16 / 5 / 8 |
| Borrowed echo, no enable | 4 / 19 / 1 / 1 | 1 / 1 | 1 / 16 / 1 / 5 |
| Borrowed echo, one enable | 8 / 26 / 2 / 2 | 2 / 1 | 2 / 16 / 2 / 7 |

The borrowed fixture's extra spectator wire is excluded from the source peak.
Full primitive inventories are recorded in the external
`g2-unary-lifetimes-resources.json` receipt. Counts and model estimates retain
their distinct decomposition levels.

Q-0597 remains open for mixed empty/nonempty axes, unresolved symbolic physical
execution and automatic foreign snapshot borrowing. The typed native borrowing
consumer is accepted above. The existing foreign snapshot manager still owns
clean-only storage and rejects `qborrow`; this batch preserves that explicit
boundary. Proposals 116, 117 and 118 remain pending.

An unchanged source conversion counterexample is retained explicitly. Running
`decompose_from_cirq_style_method` on the same echo's `qborrow` generator
produces `Allocate(..., dirty=False)`, whereas its explicit typed graph retains
`dirty=True`. The reference source is unchanged. This conversion loses the
borrowing declaration before physical binding, so typed graph and direct
Cirq-generator lifetime claims must remain distinct.

The batch passed 841 concentrated cases, 17 local integration cases and 112
cases on Cirq 1.6.1, with no skips. Strict typing of all four changed Python
files, Black, Ruff, team scope, architecture, multi-level IR and protected API
checks passed. The frozen three source blobs and nine supporting production
owners are unchanged. The dependency scan retains 666 existing findings and
zero changed-path findings. Existing single-axis, private-clean-child,
interval, root-override and notebook evidence is reused.

The next-boundary intake additionally executes the unchanged source with an
empty row, an empty column and an unresolved positive integer column extent,
each with zero or one enable bit. Source decomposition and QEC modeling both
raise AssertionError for mixed empty axes and TypeError for the unresolved
extent. Native execution rejects the corresponding inputs with its existing
explicit ValueError diagnostics. The six-case receipt is
`g2-unary-remaining-intake.json`. These are verified source limitations rather
than successful upstream execution examples. Automatic foreign snapshot dirty
borrowing remains the concrete integration gap for the next batch.

## Complete typed graph handoff, November 21

`ecosystem/cirq/_register_graph.py` now imports the actual source component or
stored CompositeBloq into an owned native Program. The source's unchanged
flattening and serialization supply graph connections, ordered typed ports and
constructor data. The existing native library decoder and resolver own
reconstruction, scheduling and call binding. Allocate/Free map directly to the
existing clean/dirty word functions. Split/Join map to their existing typed
rebindings. Other leaves use the existing Cirq leaf converter and shared interop
qubit manager. That manager also preserves source LEFT-port release and RIGHT
allocation when wrapping measured And leaves. Compiler's existing namespace
helper isolates local leaf definitions.

The converter handles each distinct live leaf once. It never imports a
constructor named in serialized data. If two unequal live leaves have the same
serialized declaration, conversion refuses the ambiguous mapping rather than
choosing one object's behavior. Missing source decompositions, unsupported
serialization and leaf conversions retain their explicit errors. No public
export, Core opcode, schema or static Cirq contract changes are introduced.
The existing private source graph and approved Program workspace contracts own
this conversion.

The original one-axis test classes cover all four frozen width/target/control
cases, including two and three controls. Both frozen three-axis shapes also
pass exact graph primitive operands, per-wire order, feedback and separate
source costs in both directions. The existing full-channel evidence for those
larger original examples is reused. The complete tutorial, interval, root-fold,
zero-selection, extra-port and resource-hook evidence above remains applicable.

For custom nested-prefix, private-clean-child and borrowed-echo subclasses,
every logical and spectator column, a dense complex superposition and every
measurement branch pass in both actual source directions. Borrowed cases also
span both unknown work values and their coherences with an external spectator.
All primitive operands and feedback match the original graph exporter. Clean
work returns to zero, each borrowed child returns its exact joint state,
allocation/free events balance and outputs retain their original physical
wires. Binding, append, JSON roundtrip and complete quantum trajectories execute
with Qualtran, Cirq and Kickmix imports explicitly blocked after conversion.
Stored typed identity graphs import without gates or lifetime events.

Measured primitive inventories and independent source models retain the four
nested/borrowed rows from the preceding batch. Original source examples compare
actual live allocation peaks against QubitCount. In the width-three/four-target
uncontrolled fixture, pruning leaves one of the two reserved clean wires unused.
Its physical reservation has nine wires while both actual live allocation and
source modeling peak at eight. This distinction is retained explicitly.

This closes Q-0597's native dynamic child and automatic typed graph handoff
obligations. Mixed empty axes and unresolved extents retain the source failures
recorded above. The source-only Cirq generator's lost dirty declaration remains
an upstream conversion counterexample. The flat named snapshot interface still
requires clean workspace. Borrowed execution uses the actual typed graph and
existing distinct borrowed pool. This is the ownership-preserving replacement
for that consumer, with normal append retained.

Final acceptance passes 849 concentrated cases, 17 local integration cases and
177 cases on Cirq 1.6.1, with zero skips. The Cirq 1.6.1 lane retains 1,224
warnings from the exercised original SDK tests. Strict typing of both new
Python files, Black, Ruff, team scope, architecture, IR and protected API checks
pass. Ten frozen source blobs and sixteen existing production owners are
unchanged. Dependency policy retains 666 whole-repository findings and zero
changed-path findings. The external `g2-register-graph-evidence.json` receipt
also records 24 measured/model resource rows and each tested source peak.
Constructor-reference leaves outside the exercised flattened primitives retain
the existing resolver's explicit factory requirement. General controlled and
classical provider breadth remains with its existing Q-0158 owner.
