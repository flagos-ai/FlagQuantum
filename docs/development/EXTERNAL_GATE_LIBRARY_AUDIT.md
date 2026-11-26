# Serialized external gate ownership audit

## Scope and ownership

This batch binds the frozen `CirqGateAsBloq` constructor in
`cirq_interop/_cirq_to_bloq.py` and its direct serialized gate, exact decomposition,
explicit matrix, scalar phase and controlled dependency paths. The reference is
Qualtran v0.7.0 commit `bcb17bb25360ed68ce3bc084645db3e929799389`. Reference code is
unchanged. Classification is `single_device_fast_path`.

`ecosystem/cirq/_component_library.py` decodes an existing ExternalGatePayload
through Cirq's existing JSON resolver. External SDK values remain local to this
boundary and a per-compilation constructor context. Returned Programs contain
owned native registers, parameters, matrices and clean-work lifetimes. Compiler
does not import external constructors or run numerical kernels.

The adapter reuses `import_cirq_register_gate` for declared decompositions and
`import_cirq_register_optree` for already-admitted primitive or explicit matrix
leaves. It does not contract an algorithm root into a dense matrix. Exact scalar
phase reuses the existing zero-port carrier recipe. Source control and target
ports remain THRU, arrays are row-major and words are MSB first.

## Direct controlled dependencies

Full resolved functions key the external gate recipes. Generic Controlled can
reconstruct only an explicitly owned child. Other children may delegate to a
supplied independent owner. Collision-free control names and bit patterns reuse
the shared signature and control-specification owners.

Negative and grouped control constructors reuse the existing ControlledViaAnd
orchestration. That owner retains source predicate computation, positive-child
call and exact measured erasure. The external gate owner supplies the positive
child through the existing SDK importer and stored control adder. Default
fixed X/Y/Z/H providers remain available. No inverse of a measured instruction
stream is invented.

Binary gate wrappers with neither a decomposition nor a unitary leaf retain a
typed bodyless declaration. They do not become executable channels implicitly.
Physical binding explicitly refuses their missing component body. Source
ResetChannel tensor refusal remains distinct from its logical declaration.

## Verification and boundaries

Standalone gate payloads compare source signatures, serialized depths zero and
one, full complex operators, exact global phase, owned work and IR round trips.
Symbolic binding is executed with source-package imports blocked. Generic,
negative and grouped external controls compare full operators and restored work.
Stored programmable arrays exercise arbitrary one- and two-qubit local rotation
and interleaver payloads through the ordinary append infrastructure.

The original Cirq JSON serializer cannot encode a `sin` expression in the gate
payload. Its TypeError is retained. A serializable symbolic parameter is used
for constructor round trips. Existing direct native parameter expressions retain
their separately accepted nonlinear execution path.

The original two-qubit arbitrary-gate array tensor path fails on non-THRU
`C[And]`. The native stored execution is compared against the original available
Cirq execution capture for that domain. This does not claim parity with the
failing generic tensor decomposition. Local matrix leaves and their synthesis
costs remain separate. Unsupported custom JSON resolvers, nonbinary physical
ports and unowned generic child controls retain explicit failures.

## Acceptance receipt, 2026-11-24

The concentrated regression passes 129 cases without skips. Related local
integration passes 145 cases. All ten unchanged original `_cirq_to_bloq_test.py`
tests pass in the existing Cirq 1.6 compatibility environment. The native payload
and stored-array regression uses the current Cirq 1.7 environment. Four frozen
interop blobs, Qualtran 0.7.0 and actual Kickmix 0.1 Simulator are verified.
Strict mypy checks both new Python files. Black, Ruff, Ecosystem scope, language,
architecture, multi-level IR and API checks pass. Dependency analysis exactly
matches the 720 existing findings.

| Payload | Actual native SDK-import primitives | Independent original QEC model |
| --- | --- | --- |
| CXPow, exponent -0.375 | Two Ry, four X, four phase, one cphase; two used wires | One And, three Cliffords, one rotation, one measurement |
| One-qubit matrix | Six phase, one Rx, four X; one used wire | One rotation |
| Two-qubit matrix | 28 phase, four Rx, ten X, three cphase, four SX; two used wires | Three And, three Cliffords, 13 rotations, three measurements |
| Scalar phase | Two X, two phase, one restored carrier | Zero QEC gates |

These measured streams use the existing SDK importer. They are not an equality
claim against the full Qualtran named-gate decomposition. In particular, native
cphase remains atomic in this path while the named CZPow owner has its explicit
And/phase/erasure decomposition. Linking named-gate Programs and their helper
closures into this payload path is the next required fidelity step. The generic
constructor coverage counter is therefore unchanged by this batch.

## Explicit adjoint and feedback replay closure — 2026-11-26

The later [named gate audit](CIRQ_NAMED_GATE_DEFINITIONS_AUDIT.md) supersedes the
historical missing named-component step above. This batch audits the complete
scoped external-gate factory file and its exact-adjoint, measured replay and
resource-atom dependencies. It adds no stable public entry, Core operation, serialization
vocabulary, global registry or numerical implementation.

| Responsibility | Authoritative owner |
| --- | --- |
| Existing JSON payload, scoped gate identity and generic control context | `ecosystem/cirq/_component_library.py` |
| Explicit serialized `Adjoint` constructor | Same scoped factory; unknown children delegate only to the supplied independent owner |
| Original decomposition followed by reversal of typed calls | Existing `_named_gate_inverse.py` and Compiler `adjoint_component_calls`; every physical leaf requires an explicit exact component pair |
| Replayed physical body after a selected numerical/measured component | `_register_import.py`; selection is disabled inside an already converted replacement body |
| Named or opaque wrapper atom pricing | `resource_keys.py`; coarse static atoms are separate from recursive decomposition prices and emitted primitives |

An actual source-exported CXPow graph reproduced a shared replay defect. When
lowering a measured native CZPow replacement, the second traversal selected its
internal H again and consumed the next source replacement, which expected two
operands. The importer now expands that already converted body without another
source-definition lookup. Outer source siblings still use their original order,
operands, control metadata, key isolation and live-wire allocation context. The
regression uses the unchanged source graph export and checks the full complex
operator, both raw histories and zero work. It adds no special-case gate name.

The explicit adjoint owner imports the ordinary gate decomposition and reverses
calls through existing exact pairs, preserving measured cleanup. It does not
invert emitted measurement instructions or substitute an inverse dense matrix.
A supplied original port signature retains scalar/array grouping. Double
adjoints retain the original forward graph. Explicit ZPow, scalar and Reset
adjoints have no original decomposition; their metadata remains a bodyless typed
declaration and physical binding refuses it. This differs from an ordinary
atomic gate's available tensor or SDK execution, which remains unchanged.

Four actually decomposable gate families (CXPow, SwapPow, one-bit matrix and
two-bit matrix) are checked in both directions against a separately exported
original graph. Exact per-wire gate dependencies, operands and feedback match
at the common representation. S/S-dagger/T/T-dagger are compared one-to-one to
P at their fixed angles: no gates are fused, deleted or reordered by that alias
conversion. Raw counts and normalized inventories are checked separately.
The source matrix graph contains one disconnected zero-port scalar. Its
frozenset tie has no portable position relative to connected calls; each complete
four-instruction native carrier is independently checked against the original
scalar angle before comparing the remaining private-work stream. This does not
ignore a phase or claim identical carrier placement. Every small measurement
history and every complete complex operator retains the absolute phase, and
all private work is zero. Native resolution, binding and ordinary append retain
owned IR with source imports blocked. Appended measured IR is checked through
explicit raw-history execution, not the unitary-only `Circuit.state` path.

### Resource layer

A source-mapped opaque one-bit unitary costs one rotation before recursive
synthesis. A two-bit matrix's decomposition includes such atoms. Recursing past
those atoms produced 25 rotations and seven Cliffords where the original mapped
graph declares 13 rotations and three Cliffords, with three Ands and three
measurements in both models. The dedicated mapped-gate cost boundary retains
that original static-atom precedence. The existing fully recursive resource
entry is unchanged; the two scopes are not interchangeable. Stabilizer-channel
static counts follow the source predicate as well, without granting a physical
body to a nonunitary declaration.

Measured primitive inventories, allocation events and original/native analytical
model categories are recorded independently in
`ftqc-reference-env/g2-external-adjoint-resources-20261126.json` beside the worktree.
Matrix leaves are numerical input atoms, not contracted algorithm roots.

### Retained direct-dependency gaps

The 28-record intake distinguishes shallow constructor serialization, immediate
source graph generation and executable source capture. Generic controls around
CXPow reach the source controlled-And leaf that the original SDK exporter cannot
execute. Conversely, successful SwapPow source control captures differ between
an explicit `Adjoint(Controlled(...))` and `Controlled(Adjoint(...))`; equal
logical operators alone do not certify those streams. Three outer-adjoint
controlled matrix/CXPow cases still require an exact child owner where the old
SDK control decomposition introduced an unnamed atom. These control-wrapper
paths remain explicit G2 gaps. This batch does not mark the constructor catalogue
or Q-0866/Q-0876 parents complete, and does not narrow their acceptance criteria.
Unapproved static contracts and unrelated Proposal 127 remain untouched.

The mapped-cost entry also distinguishes a generic inverse wrapper from the
named gate's own specialized inverse. An unchecked generic inverse around T
has no original model; native pricing returns unavailable rather than borrowing
T-dagger's cost. Generic controls with unowned child models remain unavailable.
Existing source resource generalizers retain opaque wrapper identity and their
original behavior; the explicit cost-only entry does not replace those graph
rewrites or the recursive decomposition entry.

| Ordinary explicit inverse | Actual instructions / private peak / measurements | Original model (separate) |
| --- | --- | --- |
| CXPow | 28 / 1 / 1; four T-family instructions | And 1, Clifford 3, rotation 1, measurement 1; logical peak 3 |
| SwapPow | 30 / 1 / 1; four T-family instructions | And 1, Clifford 5, rotation 1, measurement 1; logical peak 3 |
| One-bit matrix | 9 / 1 / 0, including a four-instruction scalar carrier | rotation 3; logical peak 1 versus the native two-wire carrier realization |
| Two-bit matrix | 100 / 1 / 3; 12 T-family instructions | And 3, Clifford 3, rotation 13, measurement 3; logical peak 3 |

The 24 preexisting ordinary/scalar/positive/negative/grouped-control Programs
and complete bound results are identical to published `d8635635e`. Both actual
Cirq profiles pass the 155-case concentrated batch and the 145-case local
integration suite. The final model-only boundary revision passes all 93
related gate/adjoint/model cases in both profiles. Reference and existing
protobuf warnings are retained; there are no skips. Strict typing uses real
locked source definitions, without new type ignores or dynamic base casts.
Formatting, ownership, language, architecture, IR and stable API checks are
recorded in the final batch receipt. The original reference checkout is
unchanged; source graph capture, the bodyless atomic cases and the model-only
T inverse refusal remain separate evidence.
