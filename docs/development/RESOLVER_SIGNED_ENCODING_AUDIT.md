# Signed encoding constructor disposition

## Frozen source and native ownership

The complete conversions/ones_complement_to_twos_complement.py source and its
original test file are frozen at Qualtran v0.7.0
bcb17bb25360ed68ce3bc084645db3e929799389. Four source/test/Toffoli/resolver blobs
are checked in g2-signed-constructor-intake-20261123.json. The unchanged original
tests report three passes and three explicit no-decomposition/mode skips.

| Complete definition | Native owner and boundary |
|---|---|
| SignedIntegerToTwosComplement and bitsize | Compiler formats/legacy_signed_encoding_factories.py validates the single width field through the existing integer argument owner. |
| SignedIntegerToTwosComplement.signature | Existing register types preserve LEFT x in ones-complement and RIGHT y in signed encoding, both MSB first and with the original width. |
| SignedIntegerToTwosComplement.build_call_graph | The original independent model reports bitsize-2 Toffoli calls, without declaring a primitive body. Existing signed component acceptance compares the native emitted T equivalent separately. |
| _signed_to_twos and _SIGNED_TO_TWOS | The actual ten-bit example resolves from its untouched serialized wire. Its DocSpec retains that example. |

The original root has no decomposition, numerical callback or specialized
adjoint. Its constructor resolves to a bodyless native FunctionRecord. Physical
binding of that declaration refuses. The independently accepted
bits/signed_encoding_component.py remains the owner of FlagQuantum's reversible
physical conversion through the normal circuit.append(op) entry. No reference
root primitive stream exists to compare. Direct conjunction children retain
their frozen-source operand and feedback checks.

## Execution and resources

The existing physical graph consumes its ones-complement input and returns the
same physical word tagged as signed. Temporary carries belong to the shared
And/erasure owners and return to zero. Its inverse consumes the correct image.
The full raw-word permutation retains the duplicate negative-zero case, whose
all-ones word maps to the minimum signed integer. This separately implemented
channel is distinct from the original model-only root.

| Accounting scope for n bits | Inventory |
|---|---|
| Actual primitive stream | 4(n-2) T/TDG, 7(n-2)+1 CX, n-2 measurements and resets, n-2 clean carries. |
| Independent source model | n-2 Toffoli, zero model Clifford and measurement counts. |
| Constructor declaration | Zero emitted instructions, with physical binding refused. |

The accepted component and flat suites verify complete complex amplitudes with
an entangled spectator, every small unnormalized measurement history, exact
inverse restoration, workspace zero and ordinary DynamicCircuit append. The
constructor suite reuses those component owners and independently checks the
original root ports, missing body and model. Symbolic widths remain logical
metadata. The original one-bit ones-complement guard remains an error. Original
zero-width metadata and negative Toffoli estimate are retained as diagnostic
source data, with no physical conversion claim.

## Acceptance evidence

The complete constructor and direct component closure passes 500 concentrated
cases and 145 affected local integration cases. Existing exact signed conversion
receipts are reused for complex spectator amplitudes, every applicable small
measurement history, inverse restoration and workspace cleanup. The untouched
original suite reports three passes and three original no-decomposition/mode
skips. Four frozen source blobs are verified in the intake receipt. The actual
locked Kickmix 0.1 extension and Simulator and twelve runtime source blobs were
verified in g2-sign-storage-runtime-20261123.json and reused without an
environment change.

Strict checking of both Python files, Black, Ruff, Compiler team scope, English
language, architecture, multi-level IR and public API checks pass. Dependency
policy reports the identical pre-existing 720 findings, retained separately in
g2-signed-constructor-quality-20261123.json. No whole-repository acceptance is
claimed.
