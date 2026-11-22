# Scalar physical identity acceptance

## Ownership and interface

Compiler artifacts own immutable `OperandIdentity` and `OperandKind`. Existing
operand arrays, typed register binders and control binders remain the consumers.
The exact uint32 `physical_word` and `diagnostic_text` projections reuse this
identity. `conjugated_by_h()` exchanges classical Z/X families while preserving
the payload. Boolean Z/X values use the same native immutable identity, so an
X Boolean's original plain-bool H result becomes `OperandKind.BOOLEAN`.
Quantum and register identities refuse this classical basis conversion.

Private `_operand_tags` owns the source tag masks, scalar/mixed classification,
diagnostic tag projection and execution capacity declarations. KMB and register
format validators consume the shared capacity declarations. The scalar payload
is the low 29 bits and its family is the high three bits. Register indices use
all 32 bits without a QC family tag. Array order and existing MSB-first register
binding remain intact. User execution continues through `circuit.append(op)`.

## Complete immutable source scope

The source revision is `9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453`, Apache-2.0.
All eleven original files are Git-blob verified before compiling their unchanged
C++ definitions. The retained fixture harness is a caller of those definitions.

| Parents | Complete source |
|---|---|
| K-0135 / K-0136 | bit_id.cc / bit_id.h |
| K-0140 | id_tagging.h |
| K-0141 / K-0142 | qubit_id.cc / qubit_id.h |
| K-0154 / K-0155 | register_id.cc / register_id.h |
| K-0163 / K-0164 | xbit_id.cc / xbit_id.h |
| K-0165 / K-0166 | xbool.cc / xbool.h |

`tests/unit/fixtures/generate_scalar_identities.py` compiles the five original
implementation files in a temporary directory, executes them and removes the
binary. Its corpus contains 303 observations, including all 256 unsigned-byte
tags. The original constructor, default, raw word, payload, equality, same-family
ordering, stream/string rendering, ket predicates, H conversion, masks and
capacity declarations are covered. Existing identity tests retain hash/family
separation, binding errors and mixed/register ordering behavior.

## Explicit native adaptations and boundaries

Existing repr, constructor and exception messages remain owned by FlagQuantum.
Original C++ `invalid_argument` maps to the existing native `ValueError` family
range refusal. Explicit native family arguments replace five original scalar
constructor types and their default constructors. `OperandIdentity(kind, 0)`
represents the corresponding original default.

X Boolean diagnostic output retains `XBool(false)` and `XBool(true)`. Its ket
meaning remains plus for false and minus for true. Ordinary Boolean diagnostics
use lowercase true/false. These projections remain separate from accepted gate
operand spellings and the existing repr.

Tag diagnostics use native `OperandTag` names, with an explicit one-to-one test
mapping for all eight known source labels. This records the original
`X_BOOL_VAL` and `X_BIT_ID` debug spellings. Unknown uint8 tags retain the numeric
fallback and the source scalar classifier returns false for every value above
four. Native REGISTER kind five is separate from this classifier.

Scalar constructors admit the full 29-bit payload range, including indices at
and above the separately declared 500-million execution capacity. Register
values admit uint32, independently of the one-million register execution limit.
Existing format admission limits retain their prior behavior. Python boundaries
validate original integers and reject values outside their declared domains.

## Execution and resources

Both classical signs drive the H-converted target through the actual typed
binding, normal append, lowering and KMX roundtrip. Full normalized complex
amplitudes agree with the expected coherent conditional phase. The circuit has
one logical qubit, one caller-owned classical bit and zero work qubits or
measurements. Packed execution verifies both basis lanes, classical storage,
exact phase/sign and per-lane executed operation counts.

| Resource scope | Count |
|---|---|
| Identity projections and basis metadata conversion | Zero quantum gates, measurements and work bits |
| Statically lowered comparison consumer | Four conditional X and one conditional Z |
| Executed quantum primitives per lane, false sign | Zero |
| Executed quantum primitives per lane, true sign | Four X and one Z |
| Executed classical operations per lane | One BIT_STORE0, plus one BIT_INVERT for true sign |
| Metadata records per lane | One INFO_TAG |
| Analytical quantum estimate | None for the identity value source |

Identity conversion has no measurement histories. Accepted operation-statistics
and shared erasure tests supply their independent four-history full-complex and
work-zero evidence and are rerun with this batch. No original identity gate
sequence or resource model exists to compare beyond these source value
operations. The full builder, union and interactive wrapper parents retain their
own scope. Pending Proposals 116–119 remain unimplemented.

## Acceptance results

All five unchanged original identity binding tests pass against the verified
Kickmix 0.1 native extension with Simulator present. The source corpus
regenerates byte-exact. The concentrated identity, array, word, format, phase
and shared erasure regression passes 583 cases. Existing register execution and
FTQC integration pass 121 cases. Strict mypy passes all six changed Python
files. Both identity/tag modules have 100 percent statement and branch coverage.
Black, Ruff, team scope, architecture, multi-level IR and protected API checks
pass. The dependency checker retains exactly 720 pre-existing findings, with
an identical finding set and zero new or touched-path findings.

The external acceptance receipt is
`ftqc-reference-env/g2-scalar-identity-acceptance-20261122.json`.
