# Basis-tagged operand identities

K-0263, K-0264 and K-0265 cover the locked Python ID binding, declarations
and original tests. Their Git blobs are
`44f8fea1f3936524d93cafb1e14bd0c0a4eacde7`,
`9d7402653139117d062d2e19a51ea383fd1677f8` and
`1c41fd93145001c11a8811148c229a3215d893ff`.

| Original value family | Native identity and resolved representation |
| --- | --- |
| Qubit ID | Quantum tag and physical wire index |
| Bit ID | Classical tag resolved to an existing classical coefficient |
| X bit ID | X-classical tag resolved to an X-basis target over the same classical index |
| X Boolean | X-Boolean tag, positive or negative phase state. Its Boolean value is true for the negative state. |
| Register ID | Independent register tag and numeric payload. KMX metadata retains register IDs separately from physical members. |

An additional Boolean tag represents the source's ordinary False and True
values when a caller needs tagged ordering. Equal payloads in different
families compare unequal. Equal identities hash equally. Physical and
basis literal ordering follows the source family order, then payload.
Register IDs compare within their own family. The explicit typed IDs
enforce the locked 29-bit physical width and 32-bit register width. Actual
circuit admission separately checks wires and slots against circuit bounds.
Native object representations use the compiler type names. KMX retains
its transport spelling for physical indices and register metadata.

The six original admission helpers map to broad basis-tagged operands,
Z-enable binding, X-output binding, typed Q/B handles, quantum words, and
classical slots. The broad and restricted adapters share one operand
normalizer. The Q/B-only selection retains the existing physical integer
or classical coefficient identity. The native API distinguishes Boolean
literals from integer wire indices and accepts explicit quantum identities
where a bare integer denotes a fixed-width literal. The two original X
state strings map to the existing phase targets. The X-output adapter
reinterprets an X bit as the condition over the underlying Z slot.

The source Python wrapper owns its CPython pointer checks and scalar
representation. Native identity is an immutable value with no embedded
Python object pointer, hardware allocation or gate emission. Format
metadata and emitters keep their existing schemas.

All five original tests passed with locked Kickmix 0.1. An additional
source probe verifies upper and lower range guards, cross-family order,
hash identity, and a one-wire X-classical-target comparison through
its Python entry point. The native append path matches its five
ordered conditional X/Z operations, round-trips KMX and preserves
complex amplitudes and the input classical slot for both values.
The operand binder has no analytical resource model.

Run `python -m pytest -q tests/unit/test_kickmix_operand_identity.py`.
