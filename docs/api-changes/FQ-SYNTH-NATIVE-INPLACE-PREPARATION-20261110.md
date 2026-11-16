# Native in-place real state preparation

The authorized migration of the five-category native library admits the familiar
`inplace_prepare_amplitudes(amplitudes, bound, target)` and
`inplace_prepare_state(probabilities, bound, target)` names under
`flagquantum.synth`, including wildcard imports. Both take positional-only
arguments, finite static real vectors and an existing initialized writable
complete quantum register. The vector is power-of-two sized in 2..256 and must
match the register width. The public bound must be finite and nonnegative.

These apply a specific reversible real preparation unitary, without allocating,
resetting, measuring or overwriting the target with a state. On zero input they
prepare the same state as the existing fresh-output preparation. On arbitrary
nonzero or entangled input they apply the existing Compiler real-preparation
unitary. Adjoint uses its exact inverse, including the completion on every
nonzero basis column. Probability inputs obey the same nonnegative normalized
contract as fresh prepare_state; amplitudes and their error bound use the same
existing real-preparation checks. This does not introduce complex amplitudes or
another orthogonal-completion convention.

The frontend reuses vector parsing and emits the existing
`qreg.inplace_prepare_amplitudes` effect. Existing initialization, Const/mutability,
cleanup invalidation, inverse effects, numerical lowering and budgets remain
authoritative. Plain mutable helpers and root within/adjoint composition reuse
existing static alias checks. Initial admission accepts whole registers only;
array elements, slices, reordered/cross-owner views and quantum controls of the
preparation remain refused. Fresh prepare_amplitudes/prepare_state semantics and
signatures do not change.

Acceptance requires independent full-unitary column comparison on nonzero input,
phase-weighted and entangled inputs, adjoint/within composition, matching fresh
zero preparation, invalid vectors/widths/initialization/Const/view refusal,
actual budget refusal and source-driven native examples. Fresh 216-original replay
must preserve source identities, outcomes and measured baseline resources. This
is an unreleased Compiler/frontend admission with thin Services facade exports;
no new numerical kernel, physical IR opcode, Stable Core change or release claim.
