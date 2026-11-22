# Physical operand union acceptance

## Ownership and interface before implementation

Compiler artifacts retain the existing immutable `OperandIdentity`. Private
`_operand_views` owns the nine source union domains as typed admission and
projection contexts. It returns existing identities and primitive words. Each
source wrapper maps to a context, with its own allowed families, constant
restriction, default sentinel, available explicit casts and diagnostic spelling.
Register binders reuse the shared family admission through their established
normal typed operand path. User execution remains `circuit.append(op)`.

A context carries the original static union type, which changes bool casting
and formatting even when two wrappers store the same physical word. No second
identity class or duplicate gate implementation is required. The low 29 bits
remain the payload. The high three bits remain the scalar tag. Default quantum
or classical union storage is a false-word sentinel whose diagnostic projects
to b0 even though its family is outside that union's constructor alternatives.
It is retained through the explicit default factory and raw projection path.

Source explicit casts are payload projections with no family guard. Raw uint32
projection handles all words, including source public-field mutations. Native
typed admission continues requiring canonical immutable identities. Source
consteval restrictions map to eager native ValueError at construction/admission.
Deleted numeric constructors map to typed admission that requires an identity.
Existing register entry points retain their earlier raw-int interpretation.

The frozen source revision is
`9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453`, Apache-2.0. Complete scope is
K-0137/K-0138/K-0139/K-0143/K-0144/K-0145/K-0146/K-0147/K-0148/K-0149/
K-0150/K-0151/K-0152/K-0153, with the accepted scalar identity/tag closure.
The initial source intake verifies 14 root and 11 accepted dependency blobs,
executes 226 unchanged C++ observations and checks 11 compile-time refusals.
The full evidence and acceptance disposition will follow implementation.

## Complete file disposition

| Complete source | Parent IDs | Native domain |
|---|---|---|
| bit_or_bool.h | K-0137 | CLASSICAL_BOOLEAN |
| bit_or_false.cc / .h | K-0138 / K-0139 | CLASSICAL_FALSE |
| qubit_or_bit.h | K-0143 | QUANTUM_CLASSICAL |
| qubit_or_bit_or_bool.cc / .h | K-0144 / K-0145 | Z_OPERAND |
| qubit_or_false.cc / .h | K-0146 / K-0147 | QUANTUM_FALSE |
| qubit_or_minus_state.h | K-0148 | QUANTUM_MINUS |
| qubit_or_true.h | K-0149 | QUANTUM_TRUE |
| qubit_or_xbit_or_xbool.cc / .h | K-0150 / K-0151 | X_OPERAND |
| qubit_or_xzbit_or_xzbool.cc / .h | K-0152 / K-0153 | MIXED_OPERAND |

The private `OperandDomain` enum carries a present semantic distinction. A word
such as q0 converts to true in the Z union and false in the mixed union. A
false word renders as lowercase false, uppercase False or b0 depending on the
source static union type. An identity alone cannot determine those outcomes.
The enum contains that context only. `OperandIdentity` remains the single value
representation, and domain-to-domain conversion retains the same raw word.

Existing kind/value and physical-word projections cover the source predicates,
masking, payload and tag access. Existing identity equality and order cover all
canonical scalar overloads. The new raw-word caster preserves every original
explicit cast, including source casts from a different active variant. Typed
admission validates the original constructor alternatives and constant guards.
Raw-word diagnostics preserve deliberately mutated source storage separately.

`QubitOrFalse` has a default false sentinel and no bool constructor. Its explicit
false identity is therefore refused by typed admission. The quantum/classical
union has the corresponding false default sentinel outside its constructor
alternatives. True and minus restricted unions have no default constructors.
`BitIdOrFalse` additionally accepts numeric zero through C++ consteval bool
conversion. Native construction expresses that stored value as the existing
false Boolean identity. Other numeric constructor routes remain outside native
typed admission. Source invalid constants and deleted constructors have eager
native ValueError counterparts, with existing user entry errors preserved.

## Source and execution evidence

The retained generator verifies all 14 root blobs, 11 accepted scalar blobs and
four builder/CX consumer blobs. It compiles the original value implementations,
executes 286 observations and checks 13 original compiler refusals. Constructor
traits, every original default, equality/order overloads, type predicates,
restricted constants, explicit casts, formatting, full 29-bit boundary values
and malformed source field mutations are retained in the corpus.

The pinned Kickmix 0.1 native extension and Simulator are required. The generator
records its extension digest and refuses unavailable native execution. Twenty
original builder CX paths are executed across both independent classical bits
and their self-alias. Four source Simulator basis lanes per classical setting
produce 320 exact phase/output/count observations. Native normal append is
compared to those source columns in 80 complete coherent complex-state runs.

Nineteen builder paths retain identical raw physical records, ordered operands
and classical feedback. The self-aliased classical-to-X-classical path retains
one NEG primitive with the same effective b0 guard. The source wraps it in a
redundant b0 PUSH/POP pair. The existing native predicate representation merges
the duplicated guard and emits one conditional NEG. This is the previously
accepted effective-guard convention in REGISTER_CX_BINDING.md. The test asserts
both raw streams explicitly, uses the existing source normalizer only for this
case, and records the two source metadata operations separately from primitive
counts. Source references remain unchanged.

Valid diagnostic tokens preserve original spellings. Invalid raw-word fallback
labels use native XOperand, MixedOperand and quantum/minus names. Source
`.tagged_id` diagnostic fields map to `.physical_word`. The explicit text mapping
is retained in the tests. Existing identity repr and scalar diagnostic output
retain their prior contract.

## Resource separation

| Scope | Evidence |
|---|---|
| Identity admission, default, raw cast and diagnostic | Zero quantum primitives, measurements and work bits |
| Static consumer records | Original/native opcode histograms retained independently |
| Consumer quantum primitives | Identical ordered CX, X, Z or NEG schedule and effective guards |
| Actual execution | All 320 lane observations compare source/native grouped operation counters and exact phase |
| Aliased condition metadata | Source one PUSH and one POP per lane, native zero redundant scope markers |
| Measurement and workspace | Zero consumer measurements and zero work qubits; both classical bits retain their values |
| Analytical quantum estimate | None for the original value and binding sources |

The consumer uses two logical qubits with a normalized nonreal superposition.
Complete amplitudes include global phase. Quantum output permutations and exact
packed phases are checked independently against original native Simulator
observations. Identity conversion introduces no measurement histories. Existing
shared erasure and mixed-control regressions supply their separate small-branch
amplitude and workspace evidence and are rerun with this batch. Wider builder,
array and strided-view files retain their own parent scope. Pending Proposals
116–119 remain unimplemented.

## Acceptance results

All 105 unchanged original identity and CX tests pass. The concentrated value,
binding, broadcast, condition and erasure regression passes 742 cases, and the
register/FTQC integration regression passes 121 cases. Strict mypy passes all
four changed Python files. The new operand-view implementation has 100 percent
statement and branch coverage. The complete source corpus regenerates
byte-exact. Black, Ruff, team scope, architecture, multi-level IR and protected
API checks pass. Dependency findings remain the identical 720 pre-existing
entries, with zero new or touched-path findings.

The external acceptance receipt and complete raw/static/actual count tables are
`ftqc-reference-env/g2-operand-union-acceptance-20261122.json`. Original tests,
concentrated regression, integration, strict and quality logs share that prefix.
