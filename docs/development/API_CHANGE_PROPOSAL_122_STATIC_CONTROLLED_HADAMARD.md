# Proposal 122: static controlled-Hadamard Cirq conversion

Status: explicitly approved by the user on 2026-11-25; implementation and
source-family closure evidence are recorded in `HADAMARD_FILE_AUDIT.md`.

## Complete-file requirement

The complete locked Qualtran v0.7.0 `basic_gates/hadamard.py` and
`hadamard_test.py` inventories, Q-0240 and Q-0241, remain open for one adapter
gap: static conversion rejects the existing native `ch` instruction. Atomic
H/CH, exact tensors, inverse/control behavior, named source capture, drawing,
PennyLane H and normal append are already verified. Reuse that evidence from
`HADAMARD_FILE_AUDIT.md`; do not add another gate owner or numerical kernel.

`contracts/cirq-interop-contract.toml` explicitly lists `ch` as unsupported.
`ecosystem/cirq/conversion.py` has no static CH import/export mapping. Changing
this documented rejection needs approval under root `AGENTS.md` rule 8 and
`PUBLIC_API_PROTECTION.md`, even though Core already admits the opcode.

## Proposed narrow contract

Static `from_cirq` accepts an untagged, binary, single positive-control
`cirq.H.controlled()` operation whose child is the canonical H gate
(exponent one, global shift zero). Emit one existing `Instruction("ch", ... )`
in operation-qubit order: control first, target second. Retain the existing
wire mapping, loss report and contiguous LineQubit requirements.

Static `to_cirq` maps the existing native `ch` instruction to the same
controlled-H gate on that ordered pair. No synthesis, matrix fallback, global
phase compensation, extra work, measurement or classical feedback is added.
CH is self-adjoint and retains its exact absolute phase.

This does not admit negative, multiple or nonbinary controls, noncanonical or
symbolic H powers, tagged operations, arbitrary controlled gates, MatrixGate,
new qubit types, dynamic controls or other currently unsupported opcodes.
Existing conversion behavior and failure diagnostics for those forms remain.
In particular, signed phase gates and CY require their own dispositions.

Ecosystem owns conversion immediately into existing native IR. Core gates,
schemas, public function signatures, model fields, exports and Runtime/backend
selection are unchanged. The user continues to write `circuit.append(op)`.

## Implementation and acceptance after approval

1. Extend the existing static mapping and canonical gate recognizer; do not
   create a second conversion dispatcher or gate representation.
2. Update the Cirq contract, its verifier and existing refusal expectations
   together, documenting this approved delta. Keep all other exclusions.
3. Compare the full original H/CH source file and original test disposition;
   reuse verified named capture, tensor, drawing, control and model evidence.
4. Test native-to-Cirq-to-native exact instruction and operand roundtrips,
   reversed control/target placement, spectators and complex operator columns,
   exact inverse and SDK-free append after import. Test all stated negative
   cases and preserve existing static wire/report failures.
5. Run the existing Cirq 1.6.1 and 1.7.0 conformance lanes where available,
   isolation checks, focused source tests, regression, strict typing, format,
   language, architecture, scope, IR and public API verification.
6. Separate one emitted atomic CH with zero private work from the original
   model's two rotations and four Clifford gates. No measurement branches apply
   to this static atom; measured multi-control evidence remains separate.
7. Close Q-0240/Q-0241 only after the complete-file audit proves all remaining
   criteria, then synchronize the ledger, overview, plan, NEXT and PR evidence.

Before approval, retain the existing rejection and leave both parents open.
