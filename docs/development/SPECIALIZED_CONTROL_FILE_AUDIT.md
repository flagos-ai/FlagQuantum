# Specialized control source file audit

The reference is unchanged Qualtran v0.7.0 at
`bcb17bb25360ed68ce3bc084645db3e929799389`. The native audit verifies all
three original Git blobs before compiling the source notebook fixture cells.

Current status, 2026-11-25: approved Proposal 116 target storage and Proposal 125
nonbinary target storage are now validated through the actual public callbacks.
The latest scope and remaining boundaries are recorded in the final section.
Earlier pending-116/117 statements below describe the dated 2026-11-21 batches;
116 is implemented, and approved Proposal 122 covers the old static-CH gap.

| Frozen asset | Git blob | Ledger item |
| --- | --- | --- |
| `specialized_ctrl.ipynb` | `67c4664b39c2c1043d59312e76d9453c9a294cea` | Q-0560 |
| `specialized_ctrl.py` | `98b64995010da5863218dbf93b39bbffbd600a2f` | Q-0561 |
| `specialized_ctrl_test.py` | `8ed8dd515d1646ac5d9bcbb9e0332f3060b127df` | Q-0562 |

The [native tutorial](../../examples/notebooks/specialized_control.ipynb)
uses stored typed calls and the shared Compiler selectors. A positive
specialization keeps T and T† on the target and controls the central X only.
The two-control wrapper passes both controls through, computes the predicate
on one caller-owned clean wire, calls the positive specialization and
measurement-erases the predicate. Both outcomes preserve the complete
complex state and return the work wire to zero. The native append example
checks the locked source gates, operands, per-wire dependencies and
measurement feedback, including the reset used to reuse the work wire.

The specialized adjoint selects the positive controlled body before applying
the exact stored adjoints. Its primitive stream matches the locked
`T, CNOT, T†` source, with the phases on the target alone. The generic
wrapper in the original notebook constructs successfully. Its decomposition
and call graph fail because the uncontrolled branch passes `ctrl` and
`target` to X, which requires `q`. The native generic example implements the
intended controlled phases and CNOT and has the same complex amplitudes as
the valid specialized source. The original error is reproduced separately.

| Tutorial path | Emitted primitive counts | Locked QEC model |
| --- | --- | --- |
| Two-control specialization | 6 T/T†, 7 CX, 3 H, 1 S, 1 measured CZ feedback, 1 measurement, 1 reset | 2 T, 1 And, 2 Clifford, 1 measurement; peak 4 |
| Positive specialized adjoint | 2 T/T†, 1 CX | 2 T, 1 Clifford, 0 measurements |
| Native generic adjoint | 2 controlled phases, 1 CX | Original decomposition fails before a model can be expanded |

The original tests pass 30 cases plus the original notebook. Native
checks map all 16 grouped-control cost cases and all 12 custom-port
decomposable cases. The latter preserve the two uses of the opaque
specialized child and each predicate's separate compute and erase. Original
atomic dispatch and adjoint leaf identities are checked separately. The
opaque fixtures test graph structure or declared costs. Their placeholder
bodies carry no numerical interpretation. Existing regressions cover
actual returned-adder bindings, packed control, named selection, adjoint
masks, and direction-aware fallback with executable children.

Q-0561 remains open for the specific remaining type and shape combinations
listed in the current source disposition below. Concrete scalar AutoPartition
return protocols for nine admitted data types are now verified. Earlier audits also recorded
source callback failures while appending a control with `np.concatenate`
and when the selected control port is not first. Those failure cases retain
their original classifications and source blobs.

## Complete source disposition and typed packed handoff, 2026-11-21

The complete source file continues to be the acceptance unit. The following
map reuses accepted dependencies and identifies the remaining boundaries.

| Original responsibility | Native owner and evidence |
| --- | --- |
| `_MultiControlledFromSinglyControlled` validation, signature, predicate selection, stored decomposition and call graph | `multi_controlled_component_program`, shared conjunction and control resource owners; existing specialized-component and packed-control regressions |
| Additional control dispatch and returned callback | Existing control extension and selector owners; source `np.concatenate` callback TypeError remains a separately reproduced original defect |
| `_get_ctrl_system_1bit_cv` direct specialization, missing-negative fallback, missing-positive failure and existing-control combination | `select_component_control_system` and `specialized_component_control_system`; original dictionary mutation, returned data order and failure-stage witnesses reused |
| Recursive AutoPartition unwrapping and wrapped callback | Existing typed AutoPartition owner and `test_specialized_control_typed_handoff.py`; unwrapped source return remains distinct from the wrapped callback entry |
| Public `get_ctrl_system_1bit_cv` and positive-only `get_ctrl_system_1bit_cv_from_bloqs` | Existing shared selector, current public-helper invocation, original complete tests and tutorial; no duplicate native public wrapper required |
| `SpecializeOnCtrlBit`, `_specialize_control`, selected-adjoint graph construction and returned inverse adder | Existing `adjoint_component_control_system`, exact hierarchy adjoints and all four specialization-mask regressions |
| String display | Existing source tutorial and native owned diagram dispatch; native presentation follows FlagQuantum conventions |

The new handoff scenario uses all nine already admitted three-bit data types,
namely opaque, unsigned, bounded unsigned, Montgomery, unsigned fixed point
with fractional widths -2, 0 and 3, GF(8), and degree-two polynomials over GF(2).
Each type crosses one and two AutoPartition layers, positive and negative
single-control specialization, and both exact directions. Fixed-point metadata
is covered by approved Proposal 115. Multi-bit fixed-point controls retain
their separate rejection contract.

Source and native descriptors are constructed independently. The scenario calls
the unchanged source public helper and its actual returned callback. It checks
stored call dependencies, operand identities, input-dictionary mutation, nominal
output type and physical output aliases. Exact gate parameters, order and
measurement feedback match the original wrapped source graph. JSON round-trip
and `DynamicCircuit.append(op)` use the existing bound-operation example. A
six-wire coherent state includes the control, target, both packed bystanders,
a private clean work wire and an additional entangled spectator. Both measured
histories preserve every unnormalized complex amplitude and restore the work
wire to zero. The inverse target is evaluated independently using the conjugated
phase and reversed H/phase action.

The bounded and Montgomery cases exercise raw storage, including bit encodings
outside the nominal numerical domain. They establish storage and wiring behavior.
Their external numerical value validators keep their existing contracts.

| Direction and control | Actual primitives, with native CH kept atomic | Independent original QEC model |
| --- | --- | --- |
| Forward or inverse, positive control | CH 1, H 3, T 2, T-dagger 2, CX 6, S 1, phase 1, measured CZ feedback 1, measurement 1, reset 1 | T 1, And 1, Clifford 5, rotation 2, measurement 1; peak 5 |
| Forward or inverse, negative control | Same primitives plus X 2 | T 1, And 1, Clifford 7, rotation 2, measurement 1; peak 5 |

Partition views add zero gates, work wires or measurements. Both paths require
one private clean wire and one feedback slot. The external spectator is outside
the original five-wire peak model. These actual primitive counts and analytical
QEC counts describe different granularities and remain separate.

### Accepted evidence and remaining scope

The concentrated batch passes 520 cases with zero skips, including the complete
existing control, directional, equal-width callback, exact-adjoint, AutoPartition
and unsigned fixed-point dependency suites. The two warnings originate in
unchanged source notebook strings. The local integration lane passes 29 cases.
The additional Cirq 1.6.1 lane passes all 72 typed handoffs with zero skips.
Strict typing, Black, Ruff, team scope, architecture, IR and public API checks
pass. Four frozen source blobs and the actual Qualtran 0.7.0 / Kickmix 0.1 native
Simulator runtime are reverified. The earlier 30 original tests and notebook
remain accepted evidence and are reused.

Receipts in the shared reference environment are
`g2-specialized-typed-evidence-20261121.json`,
`g2-specialized-typed-accepted-*-20261121.log`, and
`g2-specialized-typed-cirq16-20261121.log`. The JSON contains 72 directional
primitive/model rows and every source definition name and line. Initial fixture
and strict-check failures are retained as diagnostic logs. The accepted test
uses the existing explicit self-adjoint X mapping when inverting the negative
specialization.

Q-0561 remains open. This batch closes concrete scalar packed return protocols
for the nine listed data types. Arbitrary shaped specialization combinations,
symbolic physical storage, signed and empty fixed-point admission remain outside
this evidence. Existing shaped single-control refusal, direction-aware ownership,
nonleading-control callback permutation and repeated-control source TypeError
witnesses remain separately classified. Pending Proposal 116 owns the signed and
empty fixed-point extension. Pending Proposals 117 and 118 remain independent.

## Shaped callbacks and declaration boundaries, 2026-11-21

The complete source file remains the acceptance unit. The shaped handoff extends
its existing public-helper evidence to nine admitted three-bit descriptors and
five row-major shapes, scalar, (2,), (1, 2), (2, 2), and (0, 2). All four old/new
control polarities invoke the actual unchanged source callback. Source fixtures
are independent Bloq declarations using original Split, CH and Join operations.
They do not replace or modify the frozen helper implementation.

Stored dependencies, dictionary mutation, nominal outputs and physical aliases
agree. Emitted operands, parameters, order and measurement feedback match.
JSON and the existing DynamicCircuit.append(op) path preserve the program. An
independent conditional-H oracle checks every complex amplitude with arbitrary
packed bystanders and an entangled spectator. Both unnormalized measurement
branches and all four forward/self-adjoint history pairs restore clean work.
The largest finite case uses 16 physical wires. This is coherent-state evidence,
without a claim to exhaustive dense-operator construction at that size.

Actual primitives and original QEC estimates remain separate. For a scalar word
with old/new controls both zero, actual counts are X 8, H 3, T 2, T-dagger 2,
CX 6, S 1, CH 1, measurement 1, measured CZ 1 and reset 1. The original model
reports And 1, Clifford 5, rotation 2, measurement 1 and peak 6. Other control
polarities and shapes have individual rows in the evidence receipt. Each
physical callback allocates and releases one work wire and uses one feedback
slot. Array partition views add zero gates or workspace.

Symbolic widths n and 2*n+1 retain their logical source graphs and explicitly
refuse unresolved physical binding. Unknown terminal costs are unavailable.
Signed fixed-point targets retain matching logical graphs and original model
checks, while native physical storage awaits Proposal 116. An actual symbolic
array dimension fails original construction with DecomposeTypeError. Native
construction rejects it with the structured executable-array ValueError. These
are recorded exception adaptations, without physical parity claims.

The direct AutoPartition dependency now also covers whole row-major typed
arrays in ordinary and left-only mode. Its dedicated audit records that scope.
The concentrated regression passes 752 cases and local integration passes 29.
The additional Cirq 1.6.1 lane passes all 232 focused cases. Strict typing,
Black, Ruff, Compiler scope, architecture, IR and API checks pass. Four frozen
blobs and Qualtran 0.7.0 / Kickmix 0.1 native Simulator are reverified. Receipts
are g2-specialized-shaped-evidence-20261121.json,
g2-specialized-shaped-final-accepted-*-20261121.log and
g2-specialized-shaped-cirq16-20261121.log in the shared reference environment.
Two unchanged notebook SyntaxWarnings remain in the concentrated lane.

Q-0561 remains open. The finite shaped domain above is verified. Signed/empty
fixed-point physical admission, unresolved physical widths and previously
recorded original callback defects retain their separate dispositions. Pending
Proposals 116, 117 and 118 are outside this delivery. No complete parent is
newly accepted by this batch.

## Approved target-storage dependency closure, 2026-11-25

The complete 378-line locked source file and all direct control/AutoPartition
responsibilities above remain the acceptance unit. This batch completes the
previously unverified approved storage combinations in the actual source
callbacks. Existing algorithms, primitive emitters and reference source are
unchanged. Control ports remain QBit; storage labels belong to data targets.
No fixed-point or finite-field control predicate contract is expanded.

| Target profile | Completed source/native path |
| --- | --- |
| Signed QFxp(3,-2), QFxp(3,1), QFxp(3,2) | Both directions, both polarities, one/two packed layers and all five shaped callback cases |
| Unsigned QFxp(0,0), QFxp(0,-2) | Scalar and shaped zero-width ownership, all four old/new polarities, both raw erasure outcomes and self-adjoint roundtrip |
| GF(3), GF(5), GF(9) | Raw storage callbacks on two-, three- and four-bit words, with all old/new polarities and shapes |
| Degree-one polynomials over GF(3), GF(5), GF(9) | Raw storage callbacks on four-, six- and eight-bit words with the same complete shaped grid |
| Degree-zero GF(5) polynomial | Three-bit packed callback, including one/two layers and both exact directions |

The unchanged public source helper and actual returned adders are invoked.
Stored dependency calls, operand identities, dictionary mutation, nominal dtype,
shape, physical output aliases, exact gate parameters/order, measurement feedback,
JSON and ordinary circuit.append(op) are checked. CH acts on the first raw bit
of each nonempty target word. Numerical field arithmetic and external value
validation retain their existing contracts. Unused binary encodings are covered
as raw storage, including coherent superpositions.

Zero-width words remain logical bindings and array elements. They emit no CH,
while the source's predicate compute/erase is preserved. Both measurement
outcomes agree in unnormalized complex phase, private work returns to zero,
and every joint forward/self-adjoint history restores the input and spectator.
An empty array and an array of zero-width words are checked separately.

Existing dense complex-state scenarios up to 16 physical wires are retained.
Wider field/polynomial arrays additionally have an exact factorization proof:
every emitted instruction is restricted to the two controls, each word's first
bit, and the one work bit. Output aliases retain all bystander positions. The
complete active operator, every erasure history and zero work are verified from
actual primitive IR against an independent conditional tensor-H oracle. The
operator squares to identity. The untouched data factor is exactly identity,
so the channel equality extends to all joint amplitudes and arbitrary entangled
bystanders. The largest case describes 35 source/working wires plus a spectator;
this is not represented as a dense 36-wire Runtime execution. No simulation
backend or numerical execution path is added to Compiler.

The three-bit typed grid covers 14 descriptors and both exact directions at one
and two packed layers. The shaped grid covers 21 descriptors, all five shapes
and four old/new polarity combinations. Source-success and old source failures
remain separate. The full admitted storage/context scope passes without skips.

Actual primitive inventories and original QEC models remain separate. A resource
receipt independently checks 420 shaped rows against the unchanged source. The
one private work wire and one feedback slot are restored/reused between calls;
partition views add zero gates. Source QubitCount equals actual bound width
before adding the external spectator. Root model estimates are not rewritten
to match decomposed primitive counts.

Receipts use `g2-specialized-approved-storage-*20261125` in the reference
workspace. Original complete-source tests, tutorial, source callback defects
and previous storage/lifetime evidence are reused. Six frozen source/dependency
blobs, Qualtran 0.7.0 and real Kickmix 0.1 native Simulator are reverified.

Q-0561 remains open under its complete representation criteria. Signed-zero
storage awaits Proposal 124; classical view/cast and symbolic field scopes
retain their independent pending contracts. Unresolved physical extents,
original symbolic-shape construction failure, nonleading-control permutations,
repeated-control np.concatenate TypeError and invalid generic tutorial callbacks
keep their existing dispositions. No failed reference implementation is edited
or substituted, and no whole-file or catalogue acceptance count is inflated.
