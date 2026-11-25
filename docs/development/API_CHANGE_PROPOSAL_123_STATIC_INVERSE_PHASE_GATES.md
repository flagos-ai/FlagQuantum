# Proposal 123: static inverse S and T adapter roundtrips

Status: explicitly approved by the user on 2026-11-25; implementation and
complete-family acceptance evidence are recorded in `S_GATE_FILE_AUDIT.md`
and `T_GATE_FILE_AUDIT.md`.

## Complete-file requirement

The remaining gaps in complete S/T source families Q-0253/Q-0254 and
Q-0263/Q-0264 are recorded in `S_GATE_FILE_AUDIT.md` and
`T_GATE_FILE_AUDIT.md`: public inverse-phase export and matching roundtrips.
Reuse the accepted atomic matrices, source capture, factories, DocSpecs,
drawing, controlled measurement branches, resource models and tutorials.
No new numerical gate implementation is required.

Both `contracts/cirq-interop-contract.toml` and
`contracts/pennylane-interop-contract.toml` explicitly exclude the existing
`sdg` and `tdg` opcodes. The static Cirq importer already recognizes the
inverse Z powers, and its existing internal exporter has the exact constructors;
public export does not admit them. PennyLane static conversion rejects them.
Root `AGENTS.md` rule 8 and `PUBLIC_API_PROTECTION.md` require approval before
changing those documented behaviors. Proposal 122 covers CH only.

## Proposed contract

Admit only these two existing atomic phase instructions through the existing
static converters:

| Native instruction | Cirq gate | PennyLane operation |
| --- | --- | --- |
| `sdg(q)` | `cirq.S**-1` | `qml.adjoint(qml.S(wires=q))` |
| `tdg(q)` | `cirq.T**-1` | `qml.adjoint(qml.T(wires=q))` |

Each side has one binary quantum wire and no parameters. Preserve the exact
diagonal complex phase, ordered operations, source wire placement, loss report
and existing extent policy. Reimport yields the same native opcode, without
substituting a general phase gate, synthesizing primitives or discarding phase.
Retain the existing Cirq inverse-Z-power recognition policy.

PennyLane recognition is limited to the built-in one-level adjoint of the
built-in S or T atom. It does not admit arbitrary adjoints, powers, custom
operators with a matching display name, controlled wrappers or symbolic gates.
Existing forward S/T conversion remains unchanged.

No other currently unsupported opcode, matrix gate, classical control,
measurement behavior, symbolic/trainable parameter or SDK object gains
admission. No new Core opcode, public entry, schema, result field, backend,
registry or canonical external object is introduced. Ecosystem owns both SDK
boundaries; users continue to append existing native operations normally.

## Implementation and acceptance after approval

1. Extend existing static mappings and narrow PennyLane constructor/recognizer
   branches; reuse Cirq's existing inverse-phase export constructors.
2. Update both contracts, verifiers and inverse-refusal tests together. Keep
   all other exclusions and their original diagnostics.
3. Verify one-atom and H/S/S-adjoint and H/T/T-adjoint source diagrams,
   exact instruction/operand roundtrips, full complex operators, spectator
   amplitudes, inverse restoration and SDK-free append after import.
4. Test unsupported adjoints, lookalike custom operations, parameterized,
   controlled and matrix forms; preserve wire/report and dependency isolation.
5. Compare unchanged locked source/tests and reuse the complete prior source
   audit evidence. Run both certified Cirq versions and both PennyLane versions
   where available, with exact lane versions recorded. Missing lanes remain
   explicit, not silently skipped.
6. Run concentrated regression, local integration, strict typing, formatting,
   owner/language/architecture/IR/public API and SDK contract checks. Report one
   emitted phase atom with zero work/measurement separately from source QEC
   costs (one Clifford for S-family, one T for T-family).
7. Close the four parent inventories only when their actual complete-file
   acceptance conditions are proved; synchronize ledger, plan, NEXT and PR.

Before approval, preserve the static inverse-export rejections and leave all
four inventories open. This proposal does not change Proposal 116 or 121.
