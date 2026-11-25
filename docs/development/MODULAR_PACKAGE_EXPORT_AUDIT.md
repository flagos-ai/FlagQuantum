# Modular package exports and source resolver aliases

The complete Qualtran v0.7.0 `mod_arithmetic/__init__.py` is frozen at commit
`bcb17bb25360ed68ce3bc084645db3e929799389`, blob
`774aa44d6c868ebc8eaa08386ac9b318b0d898de`. It contains thirteen class exports
and no arithmetic implementation. Q-0568 was already accepted; this batch does
not count it again.

## Interfaces and ownership

| Original exports | Authoritative dependency |
| --- | --- |
| CModAdd, CModAddK, CtrlScaleModAdd, ModAdd, ModAddK | [Addition audit](MODULAR_ADDITION_FILE_AUDIT.md) |
| KaliskiModInverse | Existing `legacy_modular_inverse_factories.py` and shared modular inverse components |
| CModMulK, DirtyOutOfPlaceMontgomeryModMul, ModDbl | Existing `legacy_modular_multiplication_factories.py` and shared modular multiplication components |
| CModNeg, CModSub, ModNeg, ModSub | [Subtraction audit](MODULAR_SUBTRACTION_FILE_AUDIT.md) |

The exact frozen resolver additionally accepts four package-level names:
`qualtran.bloqs.mod_arithmetic.CModAddK`, `.ModAdd`, `.ModSub`, and `.CModSub`.
Each now points to the same factory object as its defining-module name. There is
no second arithmetic implementation, name-guessing normalization or new public
schema. Existing registry composition picks up the aliases automatically.

Words remain MSB first. ModAdd uses Montgomery x/y words with the declared
modulus. CModAddK uses positive ctrl and an unsigned x; subtraction preserves x
and updates y, with both CModSub control polarities. Constructor functions retain
the original port order, exact adjoints and shared private clean-work ownership.
An alias changes neither logical outputs nor work leases. Users keep the
existing `circuit.append(op)` execution path.

## Evidence and source defects

`tests/unit/test_legacy_modular_package_aliases.py` builds explicit legacy wire
payloads and feeds identical bytes to the untouched source reader and native
reader. Canonical payloads are independently tested. Numerical and symbolic
metadata, both subtraction control polarities, complete Program equality,
primitive operands, measurement feedback, output bindings and work leases are
compared with their existing canonical owners. The package initializer and
resolver table Git blobs and real Kickmix Simulator are required.

The unchanged source reader cannot deserialize its own canonical ModSub or
CModSub identifier: those defining-module names are absent from its resolver.
The old package aliases do deserialize. This is retained as a source failure;
the reference registry is not patched. The native reader supports both forms.

ModAdd's source wire drops Montgomery modulus metadata. Source reconstruction
recovers it from constructor arguments, whereas native signature validation
still rejects that wire for both spellings. Registering its alias does not fix
or conceal that separate boundary. Symbolic classes remain logical declarations;
ModAddK remains an atomic source-model declaration without an original primitive
body. No alias is invented for exports absent from the frozen resolver.

Complete complex amplitudes are checked for actual alias-bound CModAddK, ModSub
and both CModSub polarities, with private work zero. The new width-two fixtures use three explicit raw histories plus the existing
all-history And-channel certificate on every valid basis column; exhaustive
small-history checks remain in the reused arithmetic evidence. Previously
accepted complete arithmetic and exact-adjoint evidence is reused. Literal
primitive streams are also compared directly to the original source walker.

Resource checks scan actual T/Tdag and measurement/reset instructions separately
from the original QEC model, then verify their established relationship. No
change in gates or model cost is expected from a name alias; complete bound
records, including lifetime and feedback data, must be equal. These checks do
not turn model-only source operations into physical decompositions.

## Actual versus model resource scan

The following width-two, modulus-two rows come from alias-bound primitives.
Private leases close in every case; live peak is not pool capacity.

| Operation | Control | Actual T/Tdag | Actual CX | Actual measurements | Actual live peak | Source QEC And / Clifford / measurements | Source qubits |
| --- | --- | ---: | ---: | ---: | ---: | --- | ---: |
| CModAddK(k=1) | 1 | 44 | 119 | 11 | 13 | 11 / 84 / 11 | 13 |
| ModSub | none | 36 | 94 | 9 | 10 | 9 / 74 / 9 | 10 |
| CModSub | 0 | 52 | 137 | 13 | 13 | 13 / 104 / 13 | 13 |
| CModSub | 1 | 52 | 137 | 13 | 13 | 13 / 104 / 13 | 13 |

## Remaining work

This closes the four exact-name registration gaps, not the full addition or
subtraction source-file scope. Their existing source domain failures, symbolic
execution, signature metadata and other recorded boundaries remain in their
own audits. The constructor inventory records 182 exact Compiler format keys,
zero canonical-only aliases and 121 keys without such a provider. Registration
alone is not complete acceptance, and Ecosystem providers remain separate.
