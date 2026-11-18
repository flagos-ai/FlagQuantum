# CNOT whole-file audit

Q-0230, Q-0231 and Q-0232 cover the original controlled-NOT notebook,
implementation and tests at Qualtran v0.7.0
`bcb17bb25360ed68ce3bc084645db3e929799389`. All three Git blobs match the
frozen inventory.

## Ownership and original definitions

The native atomic body binds one control followed by one target. Both
one-bit ports are THRU. With an occupied target the operation XORs the
control into it. The base operation needs no workspace or feedback and
is self-adjoint. The same typed body is used for the example signature
and circuit. Its declared call graph terminates at the CNOT node.

| Original definition | Native disposition |
|---|---|
| Signature, atomic decomposition refusal, adjoint, example and DocSpec | Existing typed primitive plus native teaching view. The original factory signature and zero-child graph, one leaf, atomic boundary and Bell-state round trip are checked together. |
| COPY, XOR and my_tensors | Simulation owns the two exact source rank-three factors and their axis order. Existing SSA binding supplies the internal label. Previously accepted tests compare all 24 label orders, pure and channel contraction, complex phase, gradients and repeated native calls. |
| on_classical_vals | The shared Simulation callback retains addition then modulo, including source NumPy scalar arithmetic and overflow. Domain validation is separate. New tests match the complete formatted truth table and the original invalid-bit rejection. |
| get_ctrl_system and nested adder | Compiler binds the added control before the original control and target, yielding one CCX under single positive control. Other admitted patterns use the existing negative conjugation and measured predicate reducer. Existing tests compare original port bindings, dependencies, conditional corrections, all small measurement histories and clean work. |
| as_cirq_op and as_pl_op | Native Ecosystem exports are exercised from the existing CX IR. Cirq's exact gate and ordered wires match the original, with its unitary compared to the original tensor. PennyLane's native export matrix matches the original FromBloq operation in an isolated supported 0.44.1 environment. |
| wire_symbol | Shared native score retains the filled Circle and ModPlus at the original ordered gate column. The notebook uses the shared graphical signature and score data. The foreign wire callback API and generic unknown-register diagnostics remain open as original interface boundaries. |

## Whole original test disposition

All nine functions were audited. Native factorized and matrix tests cover
the tensor matrix and Cirq composite equivalence. The original and native
PennyLane comparison execute in the isolated optional environment. Bell
amplitudes and restoration, classical rejection and exact truth table,
original score glyphs, and the factory atomic assertions are covered by
the new direct checks. Existing controlled-adder tests compare the
original debug-text assertion's port order through the native bound IR
and the full source call/return map. The foreign debug-text serialization
is kept in the original test and native output is compared by structure.

The complete unchanged source run reports 10 passed and 3 skipped.
The skipped autotester modes are decomposition, decomposition counts
and dtype checks that require a decomposition. The first is the
expected atomic-gate refusal. The new direct tests explicitly assert
this boundary and the independent terminal leaf account.

## Separate counts and teaching asset

| Account | Base CNOT |
|---|---|
| Emitted primitives | 1 CX, 0 T/TDG, 0 measurements and 0 work wires |
| Independent original QEC model | 1 Clifford |
| Tensor representation | 2 rank-three factors, 16 entries, one internal edge |

The one-added-positive-control path emits one atomic CCX. Its model
contains one Toffoli, with no additional work or measurements. A
physical T count depends on the selected CCX lowering. All measurements
for larger control predicates remain in the existing measured branch
tests.

The original 10-cell notebook and native 14-cell notebook execute in an
independent kernel. The native notebook blocks reference imports. It
retains the original instance, full graphical signature and atomic call
graph, then adds the normal `circuit.append(op)` Bell check, truth table,
score glyphs and factorized tensor example.

Q-0230 closes as a complete teaching asset and Q-0232 closes as a
complete original test disposition. Q-0231 stays open for generic
foreign wire/dtype callback and exact foreign interface lifecycle
admission beyond the tested concrete exports.

## Remaining interface closure

The actual `_cnot` factory is now imported through the existing named emitter.
Both independently captured directions retain `ctrl, target` ordering, exactly
one CX, no work, no classical slots and no binding swaps. Concrete native JSON
restoration preserves the instruction and full four-by-four complex matrix.
Reversed physical bindings with an entangled spectator preserve the complete
complex vector through `circuit.append(op)` and its inverse.

Drawer now owns one `controlled_not_wire_symbol` callback shared by CNOT and
Toffoli. Both frozen sources implement the same empty title, control Circle,
target ModPlus and exact unknown-register ValueError. The complete scalar CNOT
and shaped Toffoli source scores are checked independently. The former
Toffoli-only callback module is replaced by the shared module, with its existing
consumer updated. Foreign register objects stop at Ecosystem and source tests.

The actual source Cirq CNOT and PennyLane CNOT assertions match native exports
and both imports. The original two-CX diagram and zero-input round trip pass.
Original factory/DocSpec identity, atomic refusal, self-adjoint identity,
one-leaf graph, independent one-Clifford model and peak two are checked together.
Existing complete COPY/XOR tensor and classical-value evidence is reused.
Control dispatch remains covered by the actual returned source adders, complete
small measurement histories, clean work and the independent resource models.
There are no measurements or work slots in the atom.

Q-0231 closes. The previously closed original tests and notebook remain accepted.
The original decomposition-dependent autotester skips retain the atomic refusal.
Concrete snapshot serialization passes, with foreign object serialization left
to the existing admission boundary. Frozen implementation and test blobs are
verified before these checks and remain unmodified.

The whole-file batch passed 3,326 concentrated cases with no skips. Its 28
warnings retain the paired source/native NumPy overflow behavior. Ruff found
two import-order issues after the shared callback move, which were corrected.
All three changed Python files then passed Black, Ruff and strict typing.
Compiler scope, architecture, multilevel IR and protected API checks passed.
The six CNOT/Toffoli focused checks also passed after that import-only correction.
