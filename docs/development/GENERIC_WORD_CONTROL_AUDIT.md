# Generic word control and literal THC direction recipes

The source remains Qualtran v0.7.0 at
`bcb17bb25360ed68ce3bc084645db3e929799389`. No reference file is edited.
This work follows `_infra/controlled.py`, `mcmt/specialized_ctrl.py`,
`mcmt/controlled_via_and.py`, `basic_gates/swap.py` and
`chemistry/black_boxes.py`. Their locked blobs are checked by the source fixture.

## Interfaces and ownership

| Owner | Responsibility |
|---|---|
| `component_graph._stored_entry` | Verify an existing typed entry body, including primitive leaves. `_call_entry` retains its stricter call-only traversal contract. |
| `specialized_component_control` | Invoke atomic positive children directly in the original packed-AND helper. Missing bodies remain refused. No identity wrapper is inserted around the original atomic gate. |
| `component_bookkeeping.bookkeeping_scalar_control_adder` | Adapt the existing structured bookkeeping callback; keep outer control untouched through Split/Join and allocation/free. |
| `controlled_swap_component.positive_controlled_swap_program` | Preserve generic Controlled(CSwap): outer `ctrl2`, old `ctrl`, then MSB-first x/y words. Each bit uses the shared single-control specialization and measured cleanup. |
| `multi_control_component.positive_controlled_multi_z_program` | Preserve the single merged-clause ControlledViaAnd(Z) call, with scalar outer control before the existing control array. |
| `controlled_z_word_component.positive_controlled_z_word_program` | Preserve Split/controlled-MultiControlZ/Join, with caller-owned ctrl, ctrls and system. |
| `thc_selection_control` | Bind actual imported rotation/swap/phase/flip/bookkeeping identities. Known swap and phase definitions are checked literally. Opaque rotations remain independently declared controlled leaves. |
| `WalkContext.positive_source_block_pair` | Keep the literal source constructor pair separate from the existing stored-forward-graph reverse pair. The constructor resolver selects the literal pair explicitly. |
| `ecosystem._thc_walk_component_library` | Capture the actual original THC block, reflection, control and inverse dependencies into native immutable records; retain no SDK object. |
| `ecosystem._sparse_walk_component_library` | Offer an explicit literal source profile with original RIGHT banks, narrow reflection and missing source eraser declaration; preserve the existing extended-bank profile by default. |

Caller words and controls remain THRU. Split/Join add no numerical work. The
angle allocation/free follows original lifetimes. Conjunction work belongs to
the existing predicate/specialization owners, is reused lexically, and is
measurement-erased before release. The user still appends an operation through
`circuit.append(op)`. Compiler owns stored calls, not numerical execution or
backend selection.

## Literal source discrepancy

The forward positive THC block calls specialized SelectTHC(control_val=1), whose
external control is a bystander. Its stored graph inverse has the same
specialization. The source constructor Controlled(SelectBlockEncoding).adjoint()
instead calls generic Controlled(Adjoint(SelectTHC)), threading the new control
through its fifteen source calls. Its complete native constructor path now
exists. These two source recipes are not asserted to be equal unitaries or
mutual mathematical inverses. No reference is repaired to force agreement.

Generic control supports ordinary and adjoint SELECT, both existing control
polarities, symbolic angle/word metadata, odd spin dimensions and zero/one angle
widths wherever the source root succeeds. Original zero-address/zero-system
CSwap admission failures retain typed controlled headers, not empty bodies.
Symbolic or empty physical leaf decompositions retain their actual source
boundaries. Whole literal THC primitive counts, complex amplitudes and cleanup
proofs remain unavailable at source rotation/comparator/eraser leaves.

## Evidence and resource interpretation

The shared word controls match recursive source calls, operands and complete
complex operators. Small raw measurement histories are enumerated; private work
returns to zero. Both dynamic execution strategies preserve a joint spectator
through ordinary append. Native emitted primitive streams, operands and feedback
match the unmodified original source recipes, including its seven-T Fredkin.

For width-one generic controlled CSwap, actual counts include 11 T-family,
14 CX and one measurement/reset, with one private qubit. Width two has 22
T-family, 28 CX, two measurements/resets and one reused private qubit. Source
model categories are respectively one/two CSwap, And, Clifford and measurement;
these are unexpanded model categories, not primitive instruction counts.

Generic controlled ApplyControlledZs with one original control emits four
T-family, six CX and one measurement/reset, while the original overridden source
model reports zero gates. With two original controls it emits eight T-family,
twelve CX, four X and two measurements/resets, while the source model reports
two And, three Clifford and two measurements. The original static model and its
lost negative-control information are retained, not substituted for primitives.

Concrete native source budgets composed from existing THC and compound-control
owners match original QEC categories and peaks in both directions, all existing
control polarities, and the actual 450-rank/152-spin/20-angle/10-keep example.
The small 2/4/8/8 case has modeled 11 CSwap, 225 And, 333 Clifford, 225
measurements and peak 33. No whole literal primitive inventory is inferred.
Actual large ordinary context and its data tables reuse the accepted
`g2-thc-walk-closure-*` receipt; no unchanged massive QROM is expanded again.

## Complete available sparse control dependency graph

The actual original eight-spin, 65-row sparse example matches the literal
block, reflection, ordinary walk and adjoint call/return records. The original
narrow signature is retained, without transferring RIGHT junk to extra caller
signal ports. A captured record remains usable without source SDK imports;
physical binding still refuses the original missing eraser body. The default
extended-bank profile is compared as a complete immutable context to the
published baseline, separately from the literal profile.

The five-call generic sparse root and all four controlled Majorana bodies are
now stored independently of nested Controlled(And)'s unavailable non-THRU
decomposition. Both directions, original control polarities and spin dimensions
4/6/8 retain source calls, operands and returns. Unknown computational children
still fail. The literal sparse context supplies a separate positive source
block pair, and the constructor resolver preserves both control/adjoint orders.
Nine actual 65-row constructor paths cover positive, negative and grouped
controls, with all three control/adjoint orders.

Sparse constructor contexts retain the original control value, including zero,
rather than only a boolean recording its presence. Adding a control to an
already specialized SELECT reuses the original packed-control helper; controlling
its adjoint uses the separately owned generic graph. Both existing polarities
and all three adjoint orders match source calls. Sixteen previously admitted
complete Programs remain equal to the published baseline.

Cirq 1.7's q-array CY and inverse-CY use three calls: controlled half-X,
packed-controlled CZ and controlled negative half-X. Scalar CYGate, used by the
Cirq 1.6 source conversion, instead specializes through its packed-control
helper. Ecosystem selects the actual owned gate view; Compiler imports no SDK.
Cirq 1.6's explicitly wrapped generic ControlledGate(Y) has a third, six-call
decomposition. Tests retain that boundary instead of attributing the three-call
graph to it. Both implemented CY profiles have full complex-operator and all
small raw-history proofs, including zero private work. The source atomic
controlled S/S-dagger action uses the existing exact Pauli-power leaf and
ordinary `circuit.append(op)`; it is not an empty declaration.

For the three-call CY realization, emitted primitives are seven H, two
controlled phases, two T, two T-dagger, six CX, one S, two CZ, one measurement
and one reset. It uses three logical bits and one private predicate. The
original model separately reports one And, two Clifford, two rotations and one
measurement. The scalar helper's model has one And, two Clifford, no rotations
and one measurement. Both modeled peaks are four. Whole generic sparse logical
peaks are 16/23/25 for 4/6/8 spins in either direction, matching source queries.
Its source QEC gate model still raises at Controlled(And); no zero or estimated
primitive total replaces that failure.

No full walk complex operator, private-work proof or primitive inventory is
claimed at unavailable source leaves.
