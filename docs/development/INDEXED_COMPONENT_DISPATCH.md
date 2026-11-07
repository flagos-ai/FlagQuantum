# Indexed component dispatch

G2-IDX01 owns the shared register dispatch underlying locked Qualtran v0.7.0
ApplyLthBloq and ApplyGateToLthQubit. Compiler binds ordered big-endian address
axes, concrete extents, positive quantum control wires, explicit clean work
and result slots. The callback accepts a C-order flattened index and one
positive predicate wire, and returns controlled child instructions. It must
preserve that predicate and all address/control words, and clean its own work.

The parent owns only address/predicate work. Child storage remains caller-owned
and must be declared separately by its operation. Nested address axes retain
separate bounded traversals; flattening is only used for dispatch. Reuse existing
MultiAnd, bounded traversal and measured uncompute instead of a second decoder.
All parent clean wires enter/leave zero; original inputs are preserved.

The target-position adapter supplies one Pauli/I label per flattened address
and one target wire per entry. The source ravel_multi_index order is retained.
Source padded-address behavior is preserved, not silently changed to identity.
The positive control interface does not reinterpret source control_val=0.
Any empty address axis and an uncontrolled singleton first axis are refused
before emission because the locked source does not define an executable path.
No Core opcode, public export, simulator or upstream dependency is introduced.

References: Qualtran v0.7.0 commit
bcb17bb25360ed68ce3bc084645db3e929799389,
`bloqs/multiplexers/apply_lth_bloq.py`,
`apply_gate_to_lth_target.py`, and `unary_iteration_bloq.py`.

## Local evidence and resource bases

Forty-two source configurations compare complete ordered gate dependencies,
measurement feedback, T/T-dagger count, measurement count and peak used width
against unmodified pinned source. They include multidimensional extents,
oversized address words, up to three positive controls, singleton populated
axes with nonempty words, and all I/X/Y/Z leaf families. The reserved parent
work is sum(axis widths)+control count-1. Source peak width counts only work
actually used; declared ports and the reserved clean buffer remain separate.

Twenty-four complete complex branch scenarios independently implement the
SELECT formula on the declared address domain and check every small measurement
history, inactive controls, restored inputs and clean work. Padded addresses
are compared against source streams, not an invented identity extension. The
non-Pauli callback also matches the original controlled-H/X stream and both
unnormalized branches of an arbitrary complex superposition.

The controlled-H example demonstrates the difference between equal gate flow
and synthesis estimates: both source and native emitter retain one atomic CH
and four explicit T/T-dagger gates. The source model treats CH as two rotations,
giving 26 T equivalents at its default rate. Those 26 are not an emitted T
count. For the four-entry Pauli example the source estimate and actual T count
both equal 12; there are three measured erasures, two parent clean wires and
six total quantum wires. No data or gate synthesis optimization is applied.

A source-only reproducer records that direct ApplyLthBloq(control_val=0)
emits exactly the same stream as control_val=1 and preserves an input with
control zero. The positive parent API does not advertise that source bug as
negative-control support. An empty inner address axis asserts in the source;
our validation refuses it explicitly. G2-IDX02 below covers adjoint dispatch. Symbolic extents and
specialized control wrappers remain separate G2 items.

Run `python -m examples.algorithms.indexed_component_dispatch` for the
`circuit.append(op)` example. Seventy-two new tests and the affected original
register, SELECT, block-encoding, walk and control consumers pass together:
278 tests, no skips. The source-reading helper recognizes atomic CZ/CY/CH
wrapper leaves and omits exact identity leaves; no reference source is changed.
Strict mypy covers the emitter and example; Black, Ruff, compiler scope and
architecture checks pass. This is a forward executable slice, not whole-file
acceptance for either upstream class or completion of G2.

## G2-IDX02 reverse traversal contract

Adjoint dispatch reverses each nested source segment tree, swaps its And
compute/erase activation values, and visits leaves in the reversed component
order. The existing shared one-axis adjoint performs the primitive And duality;
this extension propagates it through all axes. Multi-control prefix preparation
and erasure retain their source ordering. Classical slots remain per physical
work wire and can be reused across successive calls.

The generic callback owns child direction: with `adjoint=True`, it must emit
the controlled adjoint of the selected child. The parent must not invert an
arbitrary measured IR list. Pauli target leaves are self-adjoint. This is an
internal compiler option and changes no Core/public schema or execution backend.

Reference extraction uses the original `source.adjoint().decompose_bloq()`
graph. Calling the inherited `decompose_from_registers` stub on Adjoint is not
its executable export path. Preserve source reverse component dependencies and
map the original atomic leaves; do not replace the reference with an inverse
of the native output.


### Reverse acceptance

Forty-eight original adjoint configurations preserve ordered primitive
operands/dependencies, classical feedback, T/measurement categories and used
peak width. Coverage includes three address axes, three positive controls,
oversized address words, I/X/Y/Z leaves and both component/target-position
consumers. Six additional forward/reverse S/T cases match the original
CompositeBloq export, retaining the controlled phase leaves without synthesizing
or modifying the source. The four reference modules' Git blob hashes match the
locked CSV, including the original Adjoint implementation.

Five full complex scenarios enumerate every small inverse measurement history
and every complete forward/inverse history. They compare against the independent
negative-phase SELECT formula and the original input, respectively. Predicate
work is zero in each branch; original quantum inputs, absolute phase and reused
measurement slots are preserved. Composite export omits unused reserved source
ancillas: the test binding maps its occupied logical work onto the same physical
parent work wires, keeping unused caller storage distinct.

The runnable example is
`python -m examples.algorithms.indexed_component_round_trip`. It emits eight
T/T-dagger instructions, four unsynthesized controlled phases and two measured
erasures, with one parent clean qubit. Those phase leaves have no claimed
synthesized T count. Generic child resource-model composition and source control
specialization remain separate tasks.

Sixty-one new tests pass in the concentrated register, SELECT, block encoding,
walk and control regression: 339 tests, no skips. Strict mypy covers both shared
emitters and the example. Black, Ruff, compiler scope and architecture checks
pass. The prior forward behavior and source boundaries remain unchanged.

## G2-IDX03 resource composition contract

The model accepts dimensions and already-controlled child ComponentCost records,
not emitted IR or upstream objects. Child records must describe the selected
direction and identical THRU target-port widths. The caller declares total target
width separately from per-child target width: ApplyLth uses the full target,
whereas target-position dispatch gives each leaf one target among the full bank.

Existing CostCall edges own each child's local and recursive work. Their
bystander width includes all other parent ports and the parent prefix work live
at that particular index. Non-power-of-two tail paths can be shallower, so a
uniform maximum-parent-width charge would overestimate heterogeneous children.
Reserved work remains distinct from modeled peak occupied work. Model queries
must not construct a circuit or synthesize atomic controlled leaves.

The source declared parent categories count traversal And/erase pairs, toggle
CNOTs, optional uncontrolled X conjugation, and the outer MultiAnd prefix.
Children supply their controlled gate costs. Adjoint calls reverse the visit
order and consume child-adjoint models; parent category totals remain unchanged.
Source formula anomalies must not be repaired in this aggregation layer.

### Model composition acceptance

The cost builder folds the locked declared traversal counts, while retaining
one CostCall per controlled child in index order (reverse order for adjoints).
For N populated entries and c controls, parent And/erase pairs total N-2+c.
Outer control-prefix pairs do not introduce the traversal's extra toggle CNOT.
A populated node branches only when its midpoint lies below the axis extent;
count those active ancestors for each child to determine its live bystanders.
This accounts for smaller tail prefixes without emitting a quantum gate list
or allocating a padded table. Storage is proportional to the supplied children,
not to the full capacity of oversized address words.

Ninety-eight original-source model configurations compare all seven gate
categories, separate T/CCZ budgets, T equivalents and logical peak width. They
cover Pauli, H/S/T, singleton positive-control targets, oversized axes, nested
shapes, multiple controls, forward/adjoint direction, phase-only components
with no target bank, and heterogeneous arithmetic children. Every child model
is separately checked before its parent is aggregated. An expensive three-bit
controlled AddK placed at the first/last index gives 13/12 peak logical qubits
in both source and native models; adding the maximum parent prefix everywhere
would incorrectly report 13 for the tail case.

The two upstream frontends use different declared phase-leaf cost routes:
ApplyLthBloq controls SGate/TGate directly (rotation categories), whereas the
Cirq target-position frontend converts their controlled gates to CZPowGate
(And/erasure plus a phase leaf). Tests use the actual source converter and
independently check the matching leaf declarations. Neither the model nor the
source is rewritten to make these different call chains look identical.
The original one-bit Add Clifford anomaly is retained in child aggregation.

The executable heterogeneous arithmetic example is
`python -m examples.algorithms.indexed_component_resources`. It binds children
through `circuit.append(op)` and compares the model to explicitly paired IR:

| Quantity | Model | Emitted IR |
|---|---:|---:|
| T equivalents / T instructions | 12 | 12 |
| Measurements | 3 | 3 |
| Logical peak / reserved quantum wires | 9 | 10 |

The last row has different bases: no measured execution-memory peak is claimed.
The model predicts three CCZ magic-state units at the default conversion; it
does not claim three emitted CCZ primitives. Parent and child work remains
separately owned. Every small forward and inverse complex measurement branch
of this composition clears both work regions. Both composed gate flows match
the original source dependencies, with its parent-then-child physical pool
binding preserved. A 256-bit-address model test refuses any gate-construction
call and verifies dimension-only queries.

There are 117 new tests. The concentrated model, register, SELECT, walk and
control regression passes 472 tests without skips; the final explicit budget
assertions pass all 117 model tests. Strict mypy covers the new model and
example, and Black/Ruff/scope/architecture checks pass. Symbolic extents,
legacy cost profiles and source specialized control wrappers are not closed
by this concrete default-QEC model slice.

## G2-IDX04 source control-system composition

`indexed_component_control.py` owns concrete specialization of ApplyLthBloq's
`get_ctrl_system` chain. Its grouped `ControlRegister` input uses the existing
big-endian encoding authority; an optional `(wire, activation)` describes the
component's existing control. All controls and addresses are THRU ports. Storage
is explicitly ordered: new-control predicate ladder, conjunction with the
existing control if present, address traversal. Children keep ownership of
binding, target storage and adjoint direction. No public API/schema changes.

The original call chain is `get_ctrl_system_1bit_cv` -> either direct specialized
ApplyLthBloq or `_MultiControlledFromSinglyControlled`. More than one new control
first takes the default `ControlledViaAnd` route. When a control already exists,
its conjunction remains a separate inner And. Reuse `controlled_component_ir`
and indexed traversal for these stages; flattening all controls changes source
gate dependencies and is not allowed. Inverse traversal propagates through
`AdjointWithSpecializedCtrl`; the child supplies its adjoint.

One source defect remains explicit: direct single-bit negative specialization
without an existing control executes as positive. Both emitter and model retain
that behavior for source fidelity; this internal adapter does not certify that
case as correct negative control. Multi-bit activation values and an additional
single bit on an existing control execute the specified polarities correctly.

`indexed_component_control_resource_model.py` composes the same nested calls
using the existing `ControlPredicateCall` query and `ComponentCost` authority.
It does not emit instructions. All seven logical categories and source logical
peak match independently from primitive T/T-dagger and measurement counts.
Forty-eight untouched source graphs cover forward/inverse direction, one to
three new controls, both existing polarities, QBit-shaped and integer-word
CtrlSpec encodings. Reference export flattens the original component graph
before Cirq export so the allocator sees enclosing predicate lifetimes; no
source code or source gate logic is patched. Twenty-four full complex-state
scenarios enumerate every measurement history, including inactive controls,
and prove all owned work returns to zero. Six rejection cases fail before child
emission. Two source module blobs match the pinned inventory and installed
reference environment (`98b64995010da5863218dbf93b39bbffbd600a2f` and
`3947ddf86be37f4ff3ba58fcc316a8a64922f84f`).

Run `python -m examples.algorithms.indexed_component_controls`. The append
example selects X/Z under new control word 01 and existing control zero:

| Quantity | Logical model | Emitted IR |
|---|---:|---:|
| T equivalents / T instructions | 12 | 12 |
| Measurements | 3 | 3 |
| Peak logical qubits / reserved wires | 8 | 8 |

These are separate reports even when numerical values coincide. The example
checks runtime complex states; exhaustive branch verification belongs to tests.
The new tests total 79 and the concentrated indexed/control regression passes
343 tests without skips. Strict mypy, Black, Ruff, compiler scope and architecture
checks pass. Symbolic control shapes/extents, legacy indexed model profiles,
repeated control stacking beyond this call and whole upstream file acceptance
remain open G2 work.

## G2-IDX05 QEC cost profiles

Concrete indexed and specialized-control models now accept `legacy_shims`,
matching the locked `QECGatesCost` option. This affects declared logical costs
only. The emitter takes no cost-profile option, and no circuit is constructed
by an estimate. Children must be declared under the same profile as their
parent; the parent cannot infer or rewrite the profile of arbitrary child costs.

`control_predicate_resource_model.py` owns conversion of the existing typed
`ControlPredicateCall` query to a concrete `ComponentCost`. Controls plus prefix
junk and target occupy `2*n-1` live wires in either direction. Indexed traversal
and specialized-control models both consume this authority, removing the latter's
local conversion code. Each traversal split retains compute `(1,0)` and erase
`(1,1)` in forward direction, with directions swapped for the adjoint, plus one
explicit CNOT. Positive prefix ladders retain their separate compute/erase calls.
No duplicate inversion or measured-erasure accounting rules are introduced.

The source profiles differ deliberately: a two-bit And erase has one Clifford
in the modern profile versus three plus activation flips in the historical
profile. These are model categories, not primitive counts. The append example
now shows both results for the same emitted circuit:

| Quantity | Modern model | Historical model | Emitted IR |
|---|---:|---:|---:|
| Clifford categories | 6 | 22 | Listed separately by opcode |
| T equivalents / T instructions | 12 | 12 | 12 |
| Measurements | 3 | 3 | 3 |
| Logical peak / reserved wires | 8 | 8 | 8 |

There are 112 original model configurations: 24 owned predicate cases,32 nested
or target-position cases,48 specialized-control cases and8 phase-child cases.
All seven categories and logical peaks match the source independently in both
profiles. Phase children retain their actual frontend identity: Bloq-controlled
S/T differs from the Cirq CZPowGate path. Native typed queries declare each child
before comparing it and then its parent to the source. Source module blob
`226e6db045513112bd3e5ff5aa79d2cb8c5bf6d4` is verified in both the pinned tree
and installed environment. Neither is modified.

The new suite has117 tests; concentrated regression has465 passing without
skips, including existing executable gate-flow, complete complex measurement
branch, clean-work and runtime append tests. A refusal probe proves model queries
do not enter circuit construction, and the same emitter returns unchanged IR
before and after both queries. Strict typing, format, lint, scope and architecture
checks pass. This closes concrete indexed QEC-profile composition, not general
symbolic indexed graphs or arbitrary legacy child annotations.

## G2-IDX06 repeated single-bit control roots

`conjoined_indexed_component_ir` binds the original specialized root produced
when single-bit controls are repeatedly added. The newest control is prepended
to both wires and activation values. Two controls use And; three or more use
MultiAnd. One predicate surrounds the existing positive indexed child. Predicate
storage precedes address storage, and both enter/leave zero. The first
existing-control specialization now consumes this same emitter instead of its
own two-control composition. Shared validation checks the full live layout.
This does not collapse the separate nested predicates produced when multiple
new controls are supplied in a single CtrlSpec.

`conjoined_indexed_component_cost` composes the same owned predicate pair and
existing indexed model in either QEC profile. Child binding and adjoint models
remain caller-owned. No alternative decoder or inverse of measured primitive IR
is introduced. Both the source root and its original CompositeBloq adjoint are
read directly. Their phase leaves retain atomic controlled S/T; model rotation
synthesis is not substituted for observed T instructions.

There are56 original gate-dependency cases covering every two- and three-bit
activation pattern, two alternating four-bit patterns, Pauli/phase children and
both directions. Each root is compared in both cost profiles (112 source model
comparisons).48 independent complex-state cases enumerate all measurement
histories and restore every work bit. Five nonselfadjoint round trips enumerate
all paired histories, including the three-control case. Six invalid layouts
fail before the child callback.

A source-only reproducer isolates a remaining upstream defect: the returned
root from `_MultiControlledFromSinglyControlled.get_ctrl_system` decomposes,
but invoking its returned builder adder fails at
`np.concatenate(ctrl_soqs, in_soqs[self.ctrl_reg_name])` with TypeError. Both
added control polarities reproduce the failure. Neither reference nor native
emitter pretends that the failing builder path is certified. Root execution and
its resource graph remain independently usable. Full upstream builder parity
and whole-file acceptance remain open; no patched-source consistency claim is
made.

Run `python -m examples.algorithms.indexed_component_controls --stacked` for
`circuit.append(op)` with the three-bit activation 010. Model and emitted T
counts are12, measurements3 and peak/reserved wires8. The modern model has14
Clifford categories versus22 in the historical model. This differs from the
modern6 of the one-CtrlSpec nested source route despite the same unitary; retain
actual source call identities and counting behavior. Primitive opcode counts
are printed separately.

118 new tests and583 concentrated tests pass without skips. Strict typing,
Black/Ruff, compiler ownership and architecture checks pass. Locked source is
unchanged. Repeated multi-bit additions, symbolic sizes and the upstream builder
failure are not closed by this repeated-single-bit root slice.
