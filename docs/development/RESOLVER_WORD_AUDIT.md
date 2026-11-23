# Bitwise constructor and repetition dependency audit

## Complete source group and owners

The unchanged reference is Qualtran v0.7.0 at
`bcb17bb25360ed68ce3bc084645db3e929799389`. All seven frozen bitwise,
OnEach, X-basis and resolver source/test blobs are verified in
`ftqc-reference-env/g2-bitwise-constructor-intake-20261123.json`.
Existing whole-file protocol acceptance in `BITWISE_FILE_AUDIT.md`,
`ON_EACH_FILE_AUDIT.md` and `X_BASIS_FILE_AUDIT.md` is reused.

| Constructor | Existing semantic owner | Interchange binding |
|---|---|---|
| Xor | `registers/bits/bitwise_component.py` | Two identical typed through words x and y, MSB paired CX calls |
| XorK | The same word owner and existing Simulation scalar codec | One typed through word x, concrete constant encoded before ordered X calls |
| BitwiseNot | Existing bitwise and OnEach word owners | One typed through word x, stored OnEach child |
| OnEach dependency | `registers/bits/on_each.py` | n, referenced scalar QBit THRU child named q, optional target dtype |
| XGate dependency | Existing fixed-gate factory and native X primitive | Scalar QBit q, no constructor arguments, exact self-adjoint X |

Compiler `formats/legacy_word_factories.py` validates constructor arguments and
signatures, consumes an explicitly supplied MSB constant encoder and constructs
existing typed word bodies. Ecosystem `_component_library.py` binds this callback
to Simulation `classical_register_codec.encode_register_value`. Numerical
validation and dtype-specific encoding stay with that existing owner. No
Compiler-to-Simulation import, duplicated codec, source SDK import or public
schema is introduced. The encoder can be replaced without changing consumers.

Words and bit order remain caller-owned, including reversed physical bindings.
All word roots receive resolver-assigned names and isolated helper definitions.
Users continue to execute the resulting operation through `circuit.append(op)`.
Every original bitwise example resolves at serialized depths 0, 1 and 2.
Concrete named and directly decomposed bitwise libraries emit identical
source-comparable dependency streams.

## Value and physical semantics

Eight wire-supported dtype families are compared through serialization and
physical execution. They include QAny, QBit, QUInt, QInt, QIntOnesComp,
QMontgomeryUInt, BQUInt and unsigned QFxp. Small constant domains and both
out-of-range endpoints are compared with the unchanged source scalar encoder.
Signed and ones-complement negative constants retain different bit patterns.
Montgomery constants retain raw stored-width behavior. Bounded integer codec
validation remains distinct from stored integer width validation.

QFxp.to_bits consumes the stored integer. In particular QFxp(4,2).to_bits(1)
is 0001. An earlier external continuation note incorrectly said 0100 and was
corrected after checking the original raw receipt. Quantization is a separate
existing operation. No production codec was changed.

Symbolic constants and dtype declarations preserve source symbolic classification,
including Sympy integer literals. They produce logical declarations and refuse
finite physical binding. The two symbolic original factories retain their
logical signatures. This does not add symbolic physical execution. Native
physical type admission and the existing zero-width bitwise refusal are retained.

The frozen serializer does not support QGF dtype metadata. Xor/BitwiseNot QGF
serialization raises its original TypeError. XorK with a Python integer and QGF
can fail earlier while constructing its decomposition because the scalar
encoder requires a field element. These are source failures. They are not
repaired or presented as native quantum field constructor acceptance.

## Phase, work and resources

Full coherent complex64 and complex128 states are compared with independently
contracted source tensors on reversed data wires and an entangled spectator.
Applying each self-adjoint bitwise operation twice restores the complete state.
Produced work, measurements, resets and classical feedback are all absent.
The branch set therefore contains one deterministic history. OnEach X/H/S/T
cases verify exact complex phase, widths 0/1/3 and spectator preservation.

Primitive inventories are popcount(encoded k) X for XorK, width CX for Xor and
width X for BitwiseNot. Source QEC model Clifford counts are checked separately,
with zero T and measurement categories. The existing native analytical bitwise
and width suites are reused in the concentrated regression. Model counts do
not replace the executed primitive scan.

## Remaining boundaries

OnEach currently admits the existing self-contained scalar QBit THRU child
interface. A child function containing calls to additional helpers refuses before
namespacing because the existing referenced-argument contract supplies only its
root FunctionRecord. Generic helper-closure recovery, non-THRU/shaped children
and source constructor quirks remain explicit OnEach gaps. No generic OnEach
catalogue completion is claimed. Q-0876 remains open.

The complete bitwise constructor group adds three identifiers and the X leaf
adds one, bringing complete built-in coverage to 32 of 303. OnEach and the
previous scoped MultiAnd Adjoint are additional partial mappings within the
remaining 271 identifiers. Generic Adjoint/Controlled/Power closure, pending
Proposals 116–119, GPU, performance and whole-repository qualification remain
open.

## Batch verification

The concentrated affected regression passes 3,057 cases. Local runtime
integration passes 145 cases. The coverage selection passes 117 cases and
measures 100 percent statements and branches in legacy_word_factories.py and
_component_library.py. Strict typing covers all six changed Python files.
Black, Ruff, both team scopes, repository language, architecture, IR and public
API checks pass. The dependency checker still exits one with the same 720
findings as the recorded baseline. These findings remain unresolved.

All 41 unchanged original bitwise and OnEach tests pass, with four original
mode skips and one source xfail. Seven frozen Git blobs and the native Kickmix
0.1 extension and callable Simulator are verified in
ftqc-reference-env/g2-word-runtime-20261123.json. Full logs are retained alongside
g2-word-quality-20261123.json and g2-word-regression-20261123.log.

The concentrated run exposed an obsolete codec test expecting unsigned
fixed-point physical refusal. Its unchanged published version fails independently
in g2-word-published-codec-baseline-failure-20261123.log. Proposal 115 already
approved and implemented that admission. The test now checks the approved
unsigned binding against the original raw identity binding, while retaining the
signed fixed-point refusal. No production physical-admission code changed.

## Resolved child helpers — 2026-11-23

RESOLVER_ON_EACH_AUDIT.md supersedes the earlier self-contained-child limitation.
OnEach now references the independently resolved real child and all its helpers.
Only its own Split/Join helpers are namespaced locally. Nested X and SU2 scalar/
rotation closures execute through the normal append entry with exact phase and
clean private work. Non-QBit/shaped/directional constructor gaps and pending
fixed-point storage remain open. Catalogue completion remains 66/303 after the
separately accepted SU2 constructor, with 237 identifiers remaining.
