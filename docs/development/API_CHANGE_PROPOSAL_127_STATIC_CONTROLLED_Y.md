# Proposal 127: static controlled-Y Cirq conversion

Status: explicitly approved by the user on 2026-11-27. Implementation and
acceptance remain tracked separately.

## Complete-file requirement and existing evidence

Locked Qualtran v0.7.0 `basic_gates/y_gate.py` and `y_gate_test.py`, Q-0271 and
Q-0272, retain a concrete static-conversion gap. The existing native `cy` opcode,
exact complex operator, self-adjoint behavior, source atomic capture, typed
operands, resource model and ordinary circuit.append(op) are already covered by
`Y_GATE_FILE_AUDIT.md`. Reuse those owners and tests rather than creating a gate
implementation or numerical execution path.

`contracts/cirq-interop-contract.toml` line 101 explicitly excludes `cy` from
static conversion. `ecosystem/cirq/conversion.py` maps CH but has no CY static
mapping. Root `AGENTS.md` rule 8 and `PUBLIC_API_PROTECTION.md` require explicit
approval for this documented behavior extension. Approved Proposal 122 explicitly
reserved CY for a separate disposition, so its CH approval does not authorize
this change. Proposal 123 concerns inverse phase gates only.

The original Qualtran inverse lookup also has a version-dependent distinction
between canonical cirq.CY and ControlledGate(Y). That original failure remains
independently reproduced; this proposal does not modify the locked reference or
claim the original suite passes when it fails.

## Proposed narrow contract

Static from_cirq accepts both external representations of one canonical,
untagged, positive binary control on canonical Pauli Y:

- cirq.CY when represented by CYPowGate, exponent exactly one and global shift
  exactly zero;
- ControlledGate(Y) with exactly one binary control, active value one, and a
  canonical YPowGate child with exponent one and global shift zero.

Immediately convert either representation to the existing native CY instruction,
with operation-qubit order control first and target second. Static to_cirq
exports native CY as the corresponding canonical controlled-Y operation in the
installed supported Cirq version. Preserve existing wire mapping, report fields,
loss policy and contiguous LineQubit rules.

No matrix fallback, synthesis, global-phase alignment, additional work,
measurement or feedback is introduced. This extends no Core opcode, operator
schema, public signature, export or serialization vocabulary. Ecosystem owns
only translation to and from existing native IR.

Negative, multiple and nonbinary controls, noncanonical or symbolic Y powers,
tagged operations and arbitrary controlled gates retain their current rejection.
Existing idle-wire extent, dynamic-circuit and unsupported qubit failures remain.
Other opcodes in the exclusion list remain excluded. PennyLane conversion and
runtime execution are outside this proposal.

## Implementation and acceptance

1. Extend the existing Cirq recognizer/export table and existing contract verifier;
   add no parallel dispatcher or compatibility representation.
2. Record the reviewed CY delta in the Cirq contract and revise only its affected
   refusal expectations after approval. All unrelated negative cases remain.
3. Verify both source CY representations and both supported Cirq 1.6 and 1.7
   lanes, ordered/reversed physical bindings, entangled spectators, complete
   complex operator columns, self-adjoint restoration and native IR roundtrip.
4. Verify imported native CY still executes through circuit.append(op) when
   external SDK imports are blocked; no SDK object may cross into canonical IR.
5. Preserve negative controls, extra controls, higher-dimensional controls,
   symbolic/noncanonical powers, tags, sparse wire extent and unsupported types.
6. Reuse the complete original Y-file mapping, notebook and existing measured
   multi-control evidence. Check original source failures separately; re-audit
   all remaining criteria before marking either parent complete.
7. Run strict typing, format, Ecosystem/integration scope, language, links,
   architecture, IR/API and Cirq contract checks plus affected local regression.
   Push to #1084 after local acceptance; do not wait for CI or merge the PR.
8. Report one actual atomic CY, zero work and zero feedback separately from the
   original one-Clifford, two-qubit model. This static atom has no measurement
   branches; measured predicate evidence remains with its existing owner.

Before approval, preserve the current rejection and leave Q-0271/Q-0272 open.
