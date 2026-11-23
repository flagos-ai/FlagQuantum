# Legacy component library wire format source audit

The reference is Qualtran v0.7.0 at
`bcb17bb25360ed68ce3bc084645db3e929799389`.

| Source file | Frozen Git blob | Native disposition |
| --- | --- | --- |
| `protos/bloq.proto` | `40e467b563bf100259312c6020ea116484e91baa` | Entire schema represented in private descriptors, including both oneofs and the int32 count map |
| `protos/bloq_pb2.py` | `383b347f00bf99dde57290882b9ce0a4c3ef1a62` | All 29 reachable descriptor pairs checked and deterministic bytes verified |
| `protos/bloq_pb2.pyi` | `b3481a7aabd70722b68a91541b53a7473f53d2d4` | Frozen constructor keywords compared with native descriptors, including the nested map entry |
| `serialization/bloq.py` | `132930b2804536155f5b242dbede88518fcca459` | Argument, record, discovery, native export and executable resolution paths delivered for the admitted profile |
| `serialization/bloq_test.py` | `d423a5bca747ed4828ea7758ee8df670723d8e5f` | All 10 original cases run unchanged. Native fixtures cover argument variants and the source component libraries |
| `serialization/resolver_dict.py` | `b10adb30b493afda81fcff2e330618415df655a2` | Explicit native factory mapping accepts executable definitions. Catalogue-wide mappings remain open |

A library row owns an ID, declaration, ordered endpoint connections and
logical child counts. Library discovery follows decomposition occurrences in
their first connection order, then count dependencies and constructor
references. Depth-zero entries and predicate cutoffs retain their declarations
and constructor dependencies. Revisiting a row at positive depth upgrades its
coverage with the same ID. The recursive discovery tests compare bytes for
shared roots and varying depth and predicate. The wire map has no semantic
order. Native discovery accepts count dependencies in their declaration order.

The codec preserves scalar presence, the 13 argument variants, port array
coordinates, ordered definitions, and optional T-complexity annotations.
Compressed Cirq gate arguments are carried as bytes. An explicit factory can
interpret a payload and supply a verified native definition. An external name
alone has no constructor authority. Subcomponent arguments receive a
resolved function, with cycle and missing-ID checks. Native Program export
projects ordered call operands and returns. The caller supplies legacy names,
constructor arguments and cost declarations that Program does not store.

A decoded two-CNOT composite reaches circuit append with exact gate order,
operand order, swapped output wiring and all complex amplitudes. A decoded
predicate and its referenced adjoint use the native control owner, binding
the adjoint's constructor argument to the resolved forward function. The
source comparison uses the native measurement slot convention, which
assigns slots 0, 1 and 2 to the first, second and third measurements.
All eight histories recover the input amplitudes with the expected branch
factor and clear the three work wires. Native export reconstructs both
original connection tables. The depth-one modular exponentiation fixture
roundtrips all nine rows and 4,028 bytes. The pinned source also reconstructs
that fixture in the reference environment.

| Resolved component | Observed primitive instructions | Pinned QEC model |
| --- | --- | --- |
| Two CNOT | CX2 | Clifford2, logical width 2 |
| Predicate and erase | CX18, H9, X8, T6, Tdg6, S3, measure3, conditional CZ3, reset3 | AND3, Clifford11, measurement3, logical width 7 |

Wire decoding and traversal emit zero instructions. Generic source constructor
reflection, the full 303-entry source resolver catalogue, and executable
adaptation of arbitrary foreign gates remain open under Q-0866 and Q-0876.
Executable import currently covers explicitly registered native definitions
and finite typed composite connections. Register-encoding and symbolic
grammar limits remain with their existing owners.

The native fixed-gate mapping now supplies eight catalogue identifiers from
the complete CNOT, Hadamard, S, T, Toffoli and Y source files. It retains both
S/T directions and Toffoli control-array order through ordinary circuit append.
See `RESOLVER_FIXED_GATE_AUDIT.md` for the complete constructor dependency
disposition and separate primitive/model accounts. The five lifetime and
storage-view constructors in `RESOLVER_BOOKKEEPING_AUDIT.md` extend the combined
built-in coverage to 13 identifiers. The Identity and exchange constructor group
in `RESOLVER_EXCHANGE_AUDIT.md` adds five more, bringing coverage to 18 identifiers.
The eight basic rotation constructors in `RESOLVER_ROTATION_AUDIT.md` bring
combined coverage to 26 identifiers. The complete And/MultiAnd constructors in
`RESOLVER_CONJUNCTION_AUDIT.md` bring coverage to 28 identifiers. Their scoped
Adjoint dependency is also executable, while the generic Adjoint identifier
remains partial. The remaining 275 source catalogue identifiers and generic
wrapper closure keep Q-0876 open.

The complete bitwise constructor group and its X leaf in
`RESOLVER_WORD_AUDIT.md` bring complete coverage to 32 identifiers. Its OnEach
dependency accepts self-contained scalar children and remains partial for
generic referenced helper closure. The 271 remaining identifiers include both
that gap and generic Adjoint/Controlled/Power semantics.

Complete addition.py and negate.py constructor bindings and their direct Always
dependency are mapped in RESOLVER_ARITHMETIC_AUDIT.md. Generic control/adjoint
wire wrappers and original one-bit signed Negate refusal remain explicit.

Complete subtraction.py and multi_target_cnot.py constructor dependencies are
mapped in RESOLVER_SUBTRACTION_AUDIT.md. Original fanout atomic QEC counts and
expanded CX inventories are retained separately.

Complete controlled_addition.py constructor closure is mapped in
RESOLVER_CONTROLLED_ADDITION_AUDIT.md. Both control polarities, integer dtype
metadata, source numeric controls and loaded/carry erasure reuse existing owners.
Combined complete constructor coverage is 41/303, with 262 identifiers remaining.
Generic serialized wrappers and original signed/drawing/recursion defects remain
explicit.

Complete controlled_add_or_subtract.py constructor closure and its scoped
positive Controlled(BitwiseNot) dependency are mapped in
RESOLVER_CONDITIONAL_ADDITION_AUDIT.md. Complete built-in coverage is 42/303,
with 261 identifiers remaining. The partial Controlled mapping is excluded
from the complete identifier count.

## Comparison constructor dependency batch

RESOLVER_COMPARISON_AUDIT.md closes the fourteen comparison identifiers through
existing component owners, including their private half helper. Direct typed
sign conversion, literal multi-control X/Z, scoped Pauli ControlledViaAnd and Z/CZ
leaves preserve source operands, directional aliases and measured cleanup.
Complete catalogue coverage is 60/303, with 243 remaining. The two signed-only
conversion mappings stay partial for the unchanged source's permissive same-class
non-signed inputs. Generic wrappers and structured control dependencies remain
explicit. No original source or protected numerical admission is changed.

## Partition and grouped-predicate constructor dependency batch

RESOLVER_PARTITION_PREDICATE_AUDIT.md closes Partition, Split2, Join2 and
CtrlSpecAnd through existing layout and control owners. Proposal 120 admits
symbolic array shapes in bodyless logical declarations, with finite execution
still requiring concrete dimensions. Full gate/feedback and complex branch
checks retain original ownership and clean consumed outputs. Complete catalogue
coverage is 64/303, with 239 remaining. Original numpy.uint8/Shaped/HasLength
serialization defects and generic wrapper gaps remain explicit.

## Structured controlled-constructor orchestration

RESOLVER_CONTROLLED_AUDIT.md records the complete orchestration owner and its
positive-control provider replacement path. Fixed X/Y/Z/H children, grouped and
signed controls, nested wrappers, original examples and a stored multi-target
child preserve gate operands, feedback, complex phases and clean work. Generic
serialized child dispatch remains open, so complete catalogue coverage stays
64/303, with 239 remaining. Source CH rotation estimates remain separate from
actual primitive counts.
