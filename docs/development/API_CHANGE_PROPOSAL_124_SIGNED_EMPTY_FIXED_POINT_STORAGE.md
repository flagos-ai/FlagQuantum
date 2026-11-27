# Proposal 124: signed zero-width fixed-point quantum storage

Status: explicitly approved by the user on 2026-11-27. Implementation and
acceptance remain tracked separately. Proposal 116 is approved and its
positive signed/empty unsigned implementation passes local regression.

## Additional complete-file boundary

The remaining OnEach dtype audit found a source-valid case outside Proposal
116: `OnEach(0, Hadamard(), QFxp(0, -2, signed=True))`. Unchanged Qualtran
v0.7.0 at `bcb17bb25360ed68ce3bc084645db3e929799389` constructs it and
decomposes it into Split and Join. In a real spectator context containing H
and T, its full complex tensor and Cirq circuit execute. Native storage still
refuses the descriptor, as Proposal 116 explicitly covers only positive-width
signed words and zero-width unsigned words.

Evidence is `ftqc-reference-env/g2-signed-empty-additional-boundary-20261125.json`.
The direct zero-wire source unitary protocol separately fails in the generic
empty tensor-network contraction. The spectator comparison does not relabel
that source failure as success or change its reference implementation.

## Proposed narrow contract

Allow existing canonical signed zero-width quantum fixed-point descriptors
through the same executable storage validator. Logical type validation remains
authoritative: signed width zero requires a strictly negative concrete integer
fractional width; width-zero fractional width zero remains invalid. Canonical
parameter order, exact boolean signed flag, classical width zero and metadata
identity remain required. Split/Join retain an empty linear handle and emit no
physical primitive. A physical circuit still needs a real spectator wire.

This is storage only. Numerical value admission, codecs, arithmetic, symbolic
widths or fractions, classical storage and all unapproved control interpretations
remain unchanged. No new public operation, entry, field, opcode or serialized
vocabulary is introduced. Core owns admission; Compiler reuses existing typed
composition and binding; Simulation retains numerical meanings and errors.

## Acceptance after approval

* Compare the unchanged Split/Join graph and actual foreign constructor binding
  at multiple negative fractional widths, ordinary and exact inverse directions.
* Verify full complex operators in the real spectator context, empty ownership,
  shaped storage, creation/consumption, zero work and emitted gate counts versus
  source model estimates. Retain the standalone source tensor failure separately.
* Preserve malformed, symbolic, classical, signed-one-bit and invalid-fraction
  rejections, and existing signed/multi-bit control refusals.
* Extend the current Proposal 116 concentrated regression and strict checks;
  update both approval scopes and whole-file remaining-boundary evidence.

Root `AGENTS.md` rule 8, `Core/AGENTS.md` and `PUBLIC_API_PROTECTION.md`
require explicit approval for this additional admitted profile. Proposal 116
did not authorize it; the separate user approval above now authorizes this scope.


## Implementation

The existing executable quantum storage validator now accepts canonical signed
zero-width words after logical validation, so only negative concrete fractional
widths qualify. No descriptor, call, binding or numerical runtime is added.
Focused storage checks pass 89 cases; strict typing passes all three changed
Python files. New source comparisons cover fractional widths -1, -2 and -7,
forward and exact inverse spectator operators, shaped views, ownership creation
and consumption, and actual OnEach constructor transport at depths 0/1/2.
The source standalone empty-tensor failure remains separately recorded.
