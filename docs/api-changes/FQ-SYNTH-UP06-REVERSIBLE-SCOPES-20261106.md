# Reversible Qmod scopes and Fourier selection

This UP-06 slice targets the pinned linear combination of unitaries program.
The source sequence prepares a nonproduct selector, applies a coherent
Fourier selection and then applies the adjoint of the same preparation.

A reversible within/apply scope expands to the checked forward effects,
the apply effects and the reverse sequence of adjoints. This slice uses
already initialized storage and rejects allocation, release, output transfer
and changes to lexical bindings inside the scope. Existing arithmetic
uncomputation and binding-only scopes keep their own validation paths.

The private Compiler profile accepts `qreg.inplace_prepare_amplitudes` with
the same finite normalized real vector and nonnegative bound checks as fresh
preparation. The target must already be initialized and writable. An optional
`inverse: true` applies the adjoint of the same real RY/CX ladder. Product
preparations retain their established source expansion.

The private `qreg.control_qft` effect retains the existing one-bit control
form. A `control_value` attribute selects an unsigned basis word on a distinct
initialized register of one to eight bits and a target of one to eight bits.
An optional `inverse: true` selects the adjoint Fourier transform. The
condition is coherent and all unmatched selector branches retain identity.
The Compiler uses the existing Fourier gates, gate-control rules and negative
control conjugations.

Verification compares all original complex amplitudes against an independent
selector preparation and block-diagonal Fourier reference. An additional
projection checks the expected weighted sum on the zero selector branch.
The source license, original hash, operation bounds and all corpus rows are
checked as part of the integration batch.
