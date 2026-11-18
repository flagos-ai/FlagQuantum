# Phase-addition interop and sparse preparation binding

`ecosystem/cirq/_phase_arithmetic.py` converts the locked source
`AddIntoPhaseGrad` leaf into the existing native
`compiler/registers/phase/phase_addition.py` realization. External objects stop
at Ecosystem; the emitted native snapshots and restored execution callbacks
retain no SDK instances. All words are big endian. The input word, gradient
and optional control are logical through ports; carry/loaded-control storage
starts and ends zero. Both signs, both control polarities, precision truncation,
right shifts and no-op shifts use the same native owner.

The reference is unchanged Qualtran v0.7.0 commit
`bcb17bb25360ed68ce3bc084645db3e929799389`, phase-gradient file Git blob
`97885bc9d4c054df375854ea6d9d4d3afecb63e4`, Copyright Google LLC, Apache-2.0.
The source supplies a fixed-point classical permutation, tensor simulation and
abstract Toffoli declarations. Its `decompose_bloq()` raises because no physical
recipe is declared. Therefore this adapter realizes its value contract rather
than claiming an unchanged upstream primitive stream. For 2+2 input/gradient
bits the source model declares zero Toffoli; the native realization emits four
T/T-dagger and one measurement. These are separate resource quantities.

The adapter allocates and releases clean work through the existing interop
manager. It exports with explicit typed through-port placement, then reuses the
existing private arithmetic measurement-key namespace. User source keys cannot
alias generated keys. Truncated idle bits remain declared through ports; temporary
identity anchors used only to satisfy export placement are removed before
capture. No lossy conversion is enabled and no extra identity gates are priced.

## Exported-frame lifetime

Recursive capture now reserves every Qid in an already exported sibling frame
until that frame finishes. The original exporter can eagerly free a private flag
or angle word before its descendants are expanded; a later sibling can still
use it, and Cirq can move its preparation before an earlier disjoint high-level
operation. Reserving only the current child's operands permitted an incorrect
reuse of such a prepared flag as clean work. The two sparse fixtures detect that
error in full complex amplitudes, including with the required gradient state.
The fix uses the existing `InteropQubitManager.reserve_qubits`; it does not add
another allocator or change reference code.

All old source-root gate/feedback checks are retained. The captured Hubbard LCU
now reserves eight private names instead of six, while instruction count,
T-family count and measurement count remain unchanged. The extra reservation is
reported explicitly; source model peaks and the hand-written native Hubbard
layout are unchanged.

## Complete sparse source binding

The original `SparseStatePreparationViaRotations.from_n_coeffs(N,d,b)` declaration
cannot execute with unbound shapes. Tests resolve its immutable constructor
values to concrete indices, coefficients, domain, gradient precision and target
width. They capture both actual source directions, serialize only native IR,
restore a native register emitter, and bind it through the existing
BlackBoxPrepare callback. Users still append an operation to a Circuit.

Two fixtures cover two coefficients over four labels with a 2-bit gradient and
three coefficients over eight labels with a 3-bit gradient. Every logical basis
column, including arbitrary gradient data, is compared with the existing native
sparse implementation. Full absolute complex amplitudes, all clean work outputs,
unitarity and source-adjoint restoration are checked. Independent native resource
declarations are checked against the unchanged source model.

| Case / route | Reserved wires | Instructions | Explicit T/T-dagger | Measurements | Other unexpanded leaves |
| --- | ---: | ---: | ---: | ---: | --- |
| N=4, d=2, b=2, captured source either direction | 8 | 93 | 14 | 2 | 18 local matrices, 1 CCX, 2 Rx |
| N=4, d=2, b=2, native sparse | 8 | 110 | 20 | 5 | 2 Rx |
| N=8, d=3, b=3, captured source either direction | 12 | 503 | 60 | 12 | 90 local matrices, 12 CCX, 4 Rx |
| N=8, d=3, b=3, native sparse | 12 | 531 | 96 | 24 | 4 Rx |

The independent source models are respectively 3 And, 14 Clifford,
3 measurements, peak 8; and 3 Toffoli, 18 And, 102 Clifford,
18 measurements, peak 12. Literal equality/permutation and gradient-addition
lowerings differ from these declarations. Matrix/CCX/Rx leaves are not counted
as completed Clifford+T synthesis, and whole sparse primitive parity is not
claimed just because its complete operators match.

## Branch identity and cleanup proofs

The shared test certificate still checks clean output by default. A new explicit
`check_clean_output=False` mode proves only that every raw erasure history is the
same full channel times `2**(-events/2)`, retaining every local And, unitary and
feedback obligation. Coherent source equality circuits can conservatively enlarge
Boolean support even when their actual output is clean. In that case the sparse
binding tests independently check every logical basis column of a full branch,
including all work amplitudes, and verify the complete operator's unitarity.
Linearity then proves cleanup and complex amplitudes for all inputs and histories.
This does not waive cleanup or sample histories. A dedicated regression shows
that branch identity can hold with dirty final work, that default certification
rejects it, and that invalid erasure predicates still fail in the new mode.

The standalone arithmetic fixtures additionally exhaust every small classical
input against the original callback and project complete coherent states for
all histories when small; larger cases use the channel proof. Key-collision and
native IR round-trip checks are separate. Original singleton/partial-map defects
and SciPy container admission remain explicitly tracked under Q-0697; no parent
is closed by this dependency delivery.

Run `pytest -q tests/unit/test_foreign_phase_addition.py tests/unit/test_sparse_rotation_source_binding.py`.
For the existing SDK-free user workflow, run
`python -m examples.algorithms.sparse_rotation_preparation_catalog`.

Batch acceptance: 551 concentrated cases pass (90.25 s) after updating only the
captured LCU reserved-wire expectations; all gate, feedback, phase and cleanup
checks remain enforced. The standalone phase leaf passes 38 cases; the complete focused phase/sparse/
branch-proof suite passes 51 cases, including both resolved sparse consumers.
Both production adapters and the shared proof helper pass strict typing.
Black/Ruff, ownership, architecture, multilevel IR and protected API checks pass.
Serial focused coverage passes 51 cases (13.65 s), 82% across the three complete
modules; the adapter is 86% with defensive invalid-source guards unvisited.
CPU/representation integration passes 60 cases (62.02 s), and the existing LCU
notebook passes in an independent SDK-free kernel (3.68 s). The native sparse
append catalogue also executes successfully. The first broad run's 24 failures
were the old six-private-wire snapshot expectations; the revised exact eight-wire
checks pass while preserving every numerical and channel obligation.
