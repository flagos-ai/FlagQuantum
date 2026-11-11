# Register metadata to classical tensor codecs

Simulation owns `classical_register_codec.py`, a composition boundary between
native register metadata and the existing numerical codecs. It supplies
`decode_register_bits` and `encode_register_bits`. No codec arithmetic, value
validator, graph interpreter or registry is duplicated. The unchanged reference
is Qualtran v0.7.0 at `bcb17bb25360ed68ce3bc084645db3e929799389`, specifically
`_tensor_from_classical._bits_to_classical_reg_data` and the dtype methods in
`_infra/data_types.py`.

The dense `classical_action_tensor` entry now uses these functions when codecs
are omitted. Existing explicit callbacks retain precedence and still support
other numerical domains. The node entry retains its existing callback interface;
the same functions can be passed there. Neither entry discovers an action or
infers phase from values. Actions must remain explicitly phase-free.

## Ownership and layout

| Native encoding | Existing numerical owner | Source behavior retained |
| --- | --- | --- |
| bit | bit_encoding | Original scalar element identity and vectorized arrays |
| uint, opaque, signed | integer_encoding | MSB-first scalar words and dtype-specific arrays |
| uint with bound | bounded_integer_encoding | Exclusive bound checked after decode and before encode |
| fixed_point | fixed_point_encoding | Raw stored integer, fractional width never rescales it |
| ones_complement | ones_complement_encoding | Original negative-zero collapse and magnitude errors |
| montgomery | integer_encoding scalar/default vectorized helpers | Stored unsigned word, no residue conversion; original default array path |

Quantum and classical metadata select the same underlying numerical owner;
the declared total bit width drives codecs. Shapes decide scalar versus array
decode. Output always calls the array codec, even for scalar results, as the
source does. Input reshaping in the tensor enumerator is unchanged: shaped
classical ports still encounter the original zero-quantum-width reshape boundary.
The node entry continues to refuse classical ports before enumeration.

This boundary supports concrete widths and concrete numerical parameters from
the existing constructors. Symbolic metadata and unknown encodings
require an explicit codec; they do not silently fall back to unsigned integers.
Broader numerical metadata support does not admit new types into executable
Program IR. Core verification, physical lowering, call graphs and protected APIs
are unchanged. Users continue to append physical operations with `circuit.append(op)`.

## Verification and resource accounts

114 new cases compare the actual unmodified source codec methods and dense entry:

- 51 scalar/vector/matrix cases enumerate all small binary words for 17 quantum
  and classical descriptors, checking values, scalar types, array dtypes, output
  encodings and matching failures for excluded bounded values.
- 34 empty-array cases compare each original encode/decode path, including
  source default-vectorization errors rather than replacing them with empty data.
- 17 dense cases replace explicit codec dispatch with the defaults and compare
  complete scalar-register matrices to the source. Full-capacity bounded types
  permit all binary inputs; ones' complement retains its non-injective zero map.
- Six complex tensor-node consumers reuse the codecs and verify labels, complex
  application and the complete matrix against the independently computed table.
- Three unsupported-metadata cases reject unknown encoding, negative width and
  unknown parameter. One replacement test confirms wider metadata does not pass
  Core execution verification.
- Two stored Add hierarchies derive complete 64-by-64 action tensors through the
  new defaults and original source helper. All 32 measurement branches agree on
  full complex amplitudes, retain the 1/4 projection factor and clean both work
  bits. Existing source-exact primitive stream tests remain in regression.

The codec composition and table enumeration emit no primitives and own no
quantum work bits. The physical Add consumer emits CX40, H12, T8, Tdg8, S4,
measurement4, CZ4 and reset4 with two reused clean work bits. The independent
source QEC model reports T-count16 and measurement4; these are not substituted
for measured primitive counts. A tensor inspection result alone does not certify
a new physical decomposition for fixed-point, bounded or classical metadata.

Q-0895 remains open for source-wide codec/domain integration and original
classical/empty/phase-free boundaries. No new parent or child denominator is
introduced and G3 remains unstarted.

## Local checks

114 new cases and 3,344 concentrated regressions passed (207 existing numerical
warnings). The codec composition has 100% statement/branch coverage: 56 statements
and 32 branches. Strict typing of three Python files, Black, Ruff, Simulation
scope, architecture, internal IR and protected API checks passed. Seven reference
Git blobs match the pinned inventory. Three captured old/default/source cases
show the previously missing default-codec entry; explicit old callbacks were
already valid. Evidence is retained in
`ftqc-reference-env/g2-classical-register-codec-source.json` and verification logs.

84 CPU/tensor-network integration checks passed with two existing warnings.
Default smoke/unit result: `7 failed, 22125 passed, 300 skipped, 1037 warnings in 155.15s (0:02:35)`. All observed failure IDs belong to the seven known baseline IDs; 0 baseline IDs were not observed before the max-failure stop. This is not a green full-suite result. The comparison is retained in `ftqc-reference-env/g2-classical-register-codec-default-comparison.json`.

## Admitted binary fields

[Binary field codec composition](CLASSICAL_FIELD_CODEC.md) now delegates admitted
quantum binary descriptors to the existing resolver and FiniteFieldEncoding.
Complete polynomial identity and typed array behavior are retained. The original
dense field-output type-loss and multidimensional array failures remain explicit;
this does not certify successful dense field identity or wider Core admission.

## Shared numerical field metadata

[Finite-field metadata reconstruction](FINITE_FIELD_METADATA.md) now supports
concrete odd-characteristic and classical field declarations in numerical codecs.
The executable binary entry still enforces the original Core admission. Earlier
refusals of these numerical domains have been replaced by complete source
comparisons; source dense type-loss and invalid binary words remain failures.
