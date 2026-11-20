# Specialized control source file audit

The reference is unchanged Qualtran v0.7.0 at
`bcb17bb25360ed68ce3bc084645db3e929799389`. The native audit verifies all
three original Git blobs before compiling the source notebook fixture cells.

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
