# G2 known-size storage gaps discovered in complete-file review

The 2026-11-27 Split/Join/Partition/AutoPartition review found a genuine
source-success/native-failure profile. This is a native admission gap, not an
upstream bug or a performance/platform certificate.

Unmodified QMontgomeryUInt(3,p), QFxp(3,f) and BQUInt(3,p) can allocate and bind
three raw wires through source SDK hooks. Native physical binding rejects their
symbolic interpretation metadata even though the wire count is fixed. The
before-state is recorded in `g2-word-symbolic-metadata-intake-20261127.json` in
the external reference environment. Source transport failures/metadata losses
are separate observations, not an explanation for the native direct gap.

[Proposal 130](API_CHANGE_PROPOSAL_130_CONCRETE_WIDTH_SYMBOLIC_STORAGE_METADATA.md)
records the reviewable contract, affected owners, source refusal boundaries and
acceptance checks. No Core admission change is made before approval.

The earlier complete-file closure for Q-0311, Q-0323 and Q-0321 did not cover
this valid raw-storage profile. Those parents must be reopened until the missing
admission is approved, implemented and locally accepted. Their existing
concrete, symbolic-declaration, source-error and resource evidence remains
valid and reusable. Do not delete it or pretend it proves the missing profile.

Q-0318, Q-0325, Q-0328 and Q-0331 remain open for the same direct dependency.
The current batch separately fixes the classical-width declaration bug and
retains source mixed-domain SDK refusals. That implementation repair is useful
but cannot close a file whose supported raw storage is still missing.

This disposition follows the user's approved faithful-migration rule: source
bugs, performance and platform certificates may be listed separately after a
faithful port; a missing valid-source native execution path cannot be waived.
Continue other independent G2 work while the protected contract is pending.
