# Direct controlled register export

This audit closes the direct SDK operation and diagram dependency identified in
Q-0158/Q-0159. The reference is the complete frozen Qualtran v0.7.0
`controlled.py`, revision `bcb17bb25360ed68ce3bc084645db3e929799389`, blob
`6fabe760e793f1b39f825d76f80e92b6e5fa5c28`. Copyright 2023 Google LLC,
Apache-2.0. Source `_ControlledBase.as_cirq_op` exports the child, controls that
operation with its typed equality clause, and merges the returned ports.

## Native interface and ownership

Ecosystem `control_register_operation` receives an already exported child
operation, its output leases, native control registers and activation arrays,
and explicit qubit arrays. It uses the existing typed control-value exporter
and Cirq `controlled_by` operation. Register and value traversal is row-major,
with an MSB-first trailing bit axis. The child output arrays and control arrays
retain their identities in the returned output mapping. The input mappings are
read without mutation. The child retains its own decomposition, matrix,
exponent and diagram protocols.

`RegisterCirqGate` remains the owner of native child export. The control adapter
adds no workspace or numerical kernel. A caller requiring physical execution
can use the existing register importer and binder, followed by
`circuit.append(op)`. Framework objects remain inside Ecosystem. Compiler and
Core consume their existing native records.

Control names must be distinct, present exactly once, and disjoint from returned
child names. Control qubits must be binary, distinct and separate from child
operands. Activation shape, quantum storage and codec errors are handled by
the existing typed exporter. Missing child operations fail explicitly. Cirq
retains responsibility for a child's controllability and protocol failures.

## Original diagram and operation coverage

`test_cirq_register_control.py` covers every diagram in the original
`controlled_test.py` lines 446–492. XPow(0.25) has positive, negative and two-bit
controls. Swap(2) has two-bit controls and the original register-wise cross
labels. The former compares the original exported SDK child directly. The
latter uses an actual native RegisterCirqGate for the two-register swap.

The tests compare complete text diagrams exactly. They compare full complex
unitaries with the original direct export. Both native and original operations
then pass through the same native register importer. Every resulting imported instruction,
operand and parameter is compared, and output placement is identical. The
bound operation is appended to a native Circuit on a complete complex state
with an entangled spectator. All amplitudes agree and no workspace is present.

The source Swap diagram wrapper and its direct `as_cirq_op` may use different
SDK child representations. Diagram comparison uses the original wrapper and
matrix/imported-instruction comparison uses the direct original operation. Both paths are
explicit in the test. Further tests retain the existing native child's diagram
exponent under powers 1, 2 and -1 and execute the controlled result.

## Resource evidence

Measured counts after SDK import are recorded separately in
`ftqc-reference-env/g2-declared-controls-resources.json`. Each native observation
matches the corresponding unchanged source export observation. These counts
refer to that import/decomposition granularity. The direct control adapter
itself emits no native instruction stream. Analytical resource declarations
are covered independently in `DECLARED_CONTROLLED_RESOURCE_AUDIT.md`.

## Remaining controlled-source boundaries

This dependency does not establish full Q-0158/Q-0159 acceptance. The typed
Program admission of further physical encodings, symbolic control metadata
execution and complete classical/quantum control selection still require their
own source-file acceptance evidence. Existing source decoder and singleton
matrix-shape defects remain documented in the preceding control audits.

## Completed batch checks

The concentrated dependency suite passes 4047 cases, SDK integration passes 96,
and the focused pair passes 23. Strict typing passes all four changed Python
files. Black, Ruff, both domain scopes, architecture and IR checks pass.
Dependency analysis records 687 existing whole-repository findings and zero
findings in these changed paths. The concentrated suite retains one original
source deprecation warning. Evidence logs use the `g2-declared-controls` prefix.

| Imported operation | Observed instructions | Work events |
| --- | --- | --- |
| Positive XPow(0.25) | 2 any, 1 cphase | 0 |
| Negative XPow(0.25) | 2 X, 1 phase, 3 any, 2 CX | 0 |
| Two-control XPow(0.25) | 2 X, 3 phase, 9 any, 8 CX | 0 |
| Two-control Swap(2) | 12 X, 6 T, 90 any, 72 CX, 12 CCX, 6 Tdg, 18 phase | 0 |

Each row equals the unchanged source observation at the same SDK import
boundary. The `any` entries are opaque native matrix instructions. Elementary
Clifford+T cost requires further synthesis. These measured instruction counts
and the symbolic declaration model remain separate evidence.
