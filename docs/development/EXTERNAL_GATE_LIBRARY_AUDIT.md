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

### Constructor identity follow-up

Final intake found that the existing forward control context retains the same
SDK gate for two distinct source constructors: `CirqGateAsBloq(controlled_gate)`
and `Controlled(CirqGateAsBloq(gate))`. Only the former supplies that SDK gate's
own decomposition as the explicit generic-adjoint source. A scoped native set
now records this constructor origin. Generic control wrappers require their
own exact-adjoint owner or explicit independent fallback; they no longer borrow
the raw SDK recipe. The raw controlled scalar has its own actual original graph
and complete complex operator witness. In contrast, the generic controlled
scalar's source graph is absent. SwapPow's generic controlled graph exists but
its exact chain is still queued; refusing an unowned chain is not acceptance of
that remaining G2 scope. Ordinary forward/control Programs remain unchanged.
This follow-up tightens only the newly introduced adjoint owner, not a Core API.

Compatibility ownership is Ecosystem's locked Qualtran 0.7.0 input boundary.
It can be removed when that serialized input is retired; existing native
Programs, exact component owners, cost data and ordinary execution remain
independent of it.


### Controlled external adjoint execution owner

The scoped control context now retains the original child gate and ordered
control specification independently of its forward SDK recipe. The explicit
adjoint reverses the immediate source children before controlling each inverse.
`cirq/_controlled_gate_inverse.py` admits raw SDK child execution views and the
explicit CNOT control specialization. Other named children, nested SDK control
wrappers, unknown inverse wrappers and decomposition work outside child ports
remain unavailable. This is an internal Ecosystem conversion owner, with no
Core instruction, public entry or source-runtime dependency in the resulting
Program. Compiler continues to own register calls and binding.

For SwapPow(0.375), the three original boundaries remain visible: controlled
CNOT, controlled raw CXPow inverse, controlled CNOT. Positive CNOT specializes
to Toffoli with its original packed control port. Negative, shaped-array and
unsigned-word controls retain their source control/target ports. Four actual
source graphs compare those child signatures and all primitive operands,
parameters and matrices. Every complete complex operator column and clean-work
output agrees. Ordinary `circuit.append(op)` executes the owned IR with both
source and SDK imports disabled. These four bodies have no measurements, so
there is one applicable measurement history per body.

| Control specification | Actual source/native primitive instructions | Original resource model |
| --- | --- | --- |
| Positive scalar bit | 19: 2 CCX, 2 H, 7 phase, 8 CX | Refuses at atomic `C[And]` |
| Negative scalar bit | 66: 6 X, 4 T, 27 local matrix atoms, 24 CX, 2 T-dagger, 3 phase | Refuses at atomic `C[And]` |
| Bit array `[0, 1]` | 108: 6 X, 2 T, 45 local matrix atoms, 36 CX, 6 CCX, 2 T-dagger, 11 phase | Refuses at atomic `C[And]` |
| Unsigned two-bit word equal to 1 | Same 108 primitives, retaining packed word input | Refuses at atomic `C[And]` |

Matrix atoms above are the original SDK's local decomposition leaves, not a
matrix substituted for the algorithm. All matrix payloads are compared
separately from instruction names. Native generic-control model ownership
remains absent and is not inferred from primitive inventory. The earlier
21-instruction negative `Controlled(Adjoint(raw SwapPow))` path must not replace
the 66-instruction `Adjoint(Controlled(raw SwapPow))` path. Equal operators do
not erase this source constructor distinction.

Whole-file Q0866/Q0876 acceptance remains open: generic named-child exact
controls, nested explicit wrapper composition, missing model ownership and
original unavailable graph boundaries are still separate gaps. No new parent
is counted as accepted for this dependency delivery. The unrelated unapproved
controlled-Y proposal remains untouched.


### Named power children and nested explicit adjoints

The same immediate-child owner now admits atomic XPow, YPow, ZPow and global
phase. Positive ZPow retains the existing native measured CZPow component;
positive global phase becomes a ZPow on the control. These use their actual
specialized register signatures. Qualified generic SDK execution views keep
their own decomposition, including exact eigen-phase atoms and every explicit
local matrix payload. Rz/CRz and other unowned named controls remain refused.
The SDK eigen-atom interception mirrors the existing source snapshot protocol;
it does not replace unknown child owners with an algorithm matrix.

The per-compilation factory retains each controlled child's original gate,
control specification and explicit-adjoint direction. A second or third outer
Adjoint rebuilds the directed source child graph. It never substitutes the
ordinary forward SDK recipe for an even number of explicit adjoints. The
returned Program and ordinary appended IR require no source or SDK execution.

One-bit matrix seed17 and SwapPow(0.375) each cover positive/negative scalar,
array `[0, 1]` and unsigned-word control, with one, two and three explicit outer
adjoints. All24 actual source paths retain child port signatures, operation
operands, per-wire ordering, measurement feedback, phase and matrix payloads.
Each operator column is checked in every applicable small measurement history.
Two measured erasures in the positive matrix case yield four histories; other
cases have one. Supplied work slots restore to zero. CircuitIR encoding uses
its real matrix-aware codec rather than tensor dataclass equality.

| One-bit matrix control | Actual source/native instructions | Actual measurements/private peak | Original model, separate |
| --- | --- | --- | --- |
| Positive scalar | 48 | 2 / 1 | And 2, Clifford 2, rotation 4, measurement 2; logical peak 3 |
| Negative scalar | 29, including 9 local matrix leaves | 0 / 0 | And 2, Clifford 10, rotation 4, measurement 2; logical peak 3 |
| Bit array or unsigned word | 74, including 30 local matrix leaves | 0 / 0 | And 6, Clifford 6, rotation 4, measurement 6; logical peak 5 |

The source logical cost graph and actual SDK execution view are different
artifacts, notably for negative and grouped controls. Both are recorded; no
estimate is inferred from primitive inventory. The existing native control-cost
projection lacks controlled global phase, so the independent generic matrix
estimate remains unavailable. SwapPow's original model still fails at atomic
C[And]. Neither issue is hidden by matching complete operators.

Source matrix and SwapPow paths at all three adjoint depths are delivered;
whole-file serialization/resolver parents remain open for other named-control,
allocation and source defect boundaries. This delivery does not increase the
complete-parent count or silently authorize any pending contract.


### Controlled scalar cost projection closure

The SDK boundary now translates its global-phase coefficient key through the
existing shared `sarg`/`pi` value mathematics and
`compound_control_resource.global_phase_cost_key`. This preserves the original
principal argument, including symbolic `arg(exp(i*pi*theta))`. No new resource
call, cost formula, numerical execution path or public entry is introduced.
An uncontrolled SDK scalar still retains its original free coefficient key.
The native exponent/precision pair is not misread as an SDK coefficient.

Positive scalar control becomes a ZPow model; negative/grouped control reuses
the existing predicate call model. Source scalar identity, minus identity,
S, T, general numeric and symbolic phases compare all seven QEC categories
under positive/negative scalar and two-/three-control patterns. The native
query executes with both source and SDK imports disabled after conversion.
Existing opaque and unsupported gate rules remain unchanged outside this
explicit scalar projection.

Matrix seed17 now closes its independent declared-child QEC model at one,
two and three explicit adjoint depths across four control patterns. All12
category records match the unmodified source model. The original array and
unsigned-word two-bit patterns also match in the separate24-record resource
receipt. These are model comparisons, not new physical execution claims for
three-control Programs. Reuse the previous immutable physical proofs only for
unchanged complete Programs and bindings.

| Matrix scalar and two-bit controls | Actual primitives / measurements / private peak | Matching original/native declared QEC model |
| --- | --- | --- |
| Positive scalar | 48 / 2 / 1 | And 2, Clifford 2, rotation 4, measurement 2 |
| Negative scalar | 29 / 0 / 0 | And 2, Clifford 10, rotation 4, measurement 2 |
| Array `[0, 1]` or unsigned word equal to 1 | 74 / 0 / 0 | And 6, Clifford 6, rotation 4, measurement 6 |

Actual SDK histories and logical model histories differ by design. This
closure removes the missing scalar child model without inferring costs from
emitted instructions. Source peak3/3/5 remains separately recorded; no new
independent generic composite peak estimator is claimed. Generic controlled
SwapPow still has its original atomic C[And] model failure and remains
unpriced. The following batch closes named Rz/CRz; other exact owners and
child-private allocation remain separate execution gaps. Whole-file parent
acceptance is unchanged.

### Complete controlled diagonal dependency closure

Serialized DiagonalGate roots now retain their full immediate source graph
through positive, negative, array `[0, 1]` and unsigned-word controls, including
one, two and three explicit Adjoint constructors. The existing Compiler
`controlled_rz_program(angle, adjoint=...)` owns default positive CRz. Its
adjoint reverses all four calls; substituting CRz with a negated angle would
preserve the operator but change the source call order.

The source graph inserts its connected nodes before isolated zero-port scalar
nodes. This order is retained before control connects every child to the
control register and before an outer adjoint reverses the graph. For nondefault
controls, the original `as_cirq_op` execution view remains an SDK decomposition.
Wire-free SDK phases reuse the shared scalar primitive owner, explicitly
anchored on the first declared control bit. This prevents the typed importer's
changing live-word iteration order from selecting a different phase carrier.
No new Core instruction, control-cost formula or public entry is introduced.

Complete one-, two- and three-target diagonal roots compare immediate child
ports, each wire's primitive order, operands, parameters, matrices, feedback,
complete complex operators and private-work cleanup. Native resolution runs
without Qualtran imports; normal `Circuit.append(op)` and IR round trips run
without either source or SDK imports. The tested physical paths have no
measurements or private work, so there is one measurement history per path.
All seven independent QEC categories match the original source separately,
including three-control model-only cases. Model measurements are not claimed
as physical measurements in the SDK execution view.

| Targets | Actual positive / negative / two-bit-control primitives | Original/native declared model for positive control |
| --- | --- | --- |
| 1 | 5 / 18 / 38 | Clifford 2, rotation 3 |
| 2 | 15 / 88 / 170 | Toffoli 2, Clifford 6, rotation 7 |
| 3 | 35 / 228 / 434 | Toffoli 6, Clifford 14, rotation 15 |

All three adjoint depths have matching source/native primitive inventories.
Private work is zero. Negative and grouped QEC models remain separately
recorded rather than inferred from these inventories. Existing 24 complete
Programs and physical bindings are unchanged from published `439df3f48`.

The unchanged zero-target DiagonalGate fails in the original SDK Gray-code
decomposition with `OverflowError`, in both control polarities. Native
resolution preserves that failure. Raw X/S wrappers still have no declared
decomposition; generic ZZ/SwapPow paths retain original atomic C[And]
export/model failures. Unqualified Rx/Ry children, other exact named controls
and child-private allocation are not admitted by this batch. Serialization
and resolver whole-file parents remain open; no partial closure is counted.

The concentrated batch passes 301 cases in each actual Cirq 1.7.0/Qiskit 2.5.0
and Cirq 1.6.1/Qiskit 2.0.0 environment, without skips. Local integration passes
145 cases. Three Python files pass strict typing, Black/Ruff, owner scopes,
language, links, architecture, IR and API checks. Frozen source assets,
Qualtran 0.7.0 and real Kickmix 0.1 Simulator are asserted before testing.

### Complete controlled ISwapPow dependency closure

The next independent module admits the existing named Hadamard and S/T phase
children. Default positive Hadamard reuses the native atomic CHadamard owner;
negative and grouped controls retain the source's direct SDK execution view.
The source Hadamard adjoint returns itself. Cirq HPow(-1) has the same unitary
but a different decomposition, so the adapter preserves the named source
Hadamard before controlling it. SDK scalar leaves reuse the same first-input
carrier binding as the diagonal batch. No new gate or compiler kernel is added.

Complete ISwapPow roots at exponents 0.375, 0.5, 1 and 0 cover general, T, S
and zero-angle child dispatch. Both control polarities, array and word layouts
and three explicit adjoint depths retain all eight source child calls, exact
port groupings, primitive order, operands, feedback and matrices. Complete
complex operator columns and every small raw measurement history agree, with
private work returned to zero. Normal append and serialization reuse the
existing native execution path. Negated angles are exercised by the adjoints.

| Exponent | Actual positive / negative / two-bit-control primitives | Actual positive measurements / private work | Independent positive QEC model |
| --- | --- | --- | --- |
| 0.375 | 42 / 146 / 290 | 2 / 1 | Toffoli 4, And 2, Clifford 10, rotation 6, measurement 2 |
| 0.5 or 1 | 8 / 146 / 290 | 0 / 0 | Toffoli 4, Clifford 8, rotation 6 |
| 0 | 42 / 138 / 266 | 2 / 1 | Toffoli 4, And 2, Clifford 12, rotation 4, measurement 2 |

The zero-angle source graph is deliberately retained, including its measured
And pairs; it is not replaced by an optimized identity. All 48 physical/model
receipts compare source and native inventories and all seven QEC categories
separately. Negative/grouped SDK paths have no physical measurements or private
work, even where the source logical model estimates measured And cleanup.
Three-control resource tests remain model-only. Matrices, rotations and
Toffoli remain named instructions rather than a claimed Clifford+T synthesis.

Other named fixed children, nested controlled SDK children and child-private
allocation remain explicit gaps. In particular, the source-success QFT,
PauliStringPhasor and CCXPow roots require their remaining direct dependencies.
The original FSim and generic ZZ/SwapPow C[And] failures are not waived.

The symbolic positive-control constructor is also executable. Binding the
original symbolic graph to 0.375, 0.5 or 0 preserves both measured erasures;
it does not re-run concrete S/T constructor dispatch at bind time. Native and
original captured symbolic graphs match primitive streams after binding,
complete complex branch amplitudes and clean work, with source/SDK imports
disabled during native append and binding.

The batch passes 397 concentrated cases plus three symbolic cases in each
actual SDK environment, and 145 local integration cases. Strict typing,
formatting, scopes, language, links, architecture, IR and API checks pass.
All 24 existing complete Programs and bindings match published `3fe0c1fe9`.

### Controlled CCXPow root and nested declaration costs

CCXPow numeric roots now have full execution and resource evidence across
general 0.375, half, integer and zero exponents; positive, negative, array and
word controls; and three explicit adjoint depths. They reuse the previous
Hadamard owner and existing raw SDK CCZ execution view. Each root retains its
three source children, their ports and literal primitive streams. Complete
complex operators, restored work, source-free resolution and SDK-free append
pass. These physical paths have no measurements or private work.

The remaining model gap was the raw CCZ child's declared sequence. The SDK
resource mapper now reuses the existing sequence and predicate call types:
positive control distributes over that child's declared calls, while a
nondefault activation is reduced once around the resulting positive body.
Reducing a negative activation at every nested call would overcount Cliffords;
it is not the original ControlledViaAnd call chain. Unknown child models still
decline. Compiler remains the owner of all QEC arithmetic. No resource schema,
public API or numerical execution path changes.

| Exponent | Actual positive / negative / two-bit-control primitives | Independent positive QEC model |
| --- | --- | --- |
| 0.375 or 0.5 | 36 / 78 / 182 | Toffoli 8, And 7, Clifford 15, rotation 11, measurement 7 |
| 1 | 36 / 78 / 182 | Toffoli 8, Clifford 8, rotation 11 |
| 0 | 10 / 52 / 142 | Toffoli 8, And 7, Clifford 22, rotation 4, measurement 7 |

All 48 physical/model receipts match the original inventories and seven QEC
categories separately. The model may contain measured And cleanup while the
actual SDK path has none. Three-control costs are model-only. No independent
generic composite peak estimator or fully synthesized Clifford+T count is
inferred from these results.

The original symbolic CCXPow graph and QEC model are available, and all seven
native model categories match under four activation patterns and three adjoint
depths. The original symbolic multi-controlled phase cannot be captured into
executable primitives. Native resolution retains an explicit unsupported
operation error. This is recorded as unavailable source execution, not an
accepted physical symbolic path or a waiver of the remaining parent scope.

Existing 24 complete Programs and bindings match published `2841843b9`.
QFT and PauliStringPhasor nested controlled children, other exact named controls
and child-private allocations remain open, along with the previously recorded
source failures. Whole-file serialization/resolver acceptance is unchanged.

The concentrated batch passes 509 cases without skips in each actual
Cirq 1.7.0/Qiskit 2.5.0 and Cirq 1.6.1/Qiskit 2.0.0 environment. Local
integration passes 145 cases. Three Python files pass strict typing,
Black/Ruff, owner scopes, language, links, architecture, IR and API checks.

### QFT roots and nested raw SDK control ports

Explicit serialized Adjoint/Controlled QFT roots now preserve the source's
complete child graph at zero through three targets, both output-reversal
settings represented by the two-target cases, four control layouts and three
adjoint depths. Nested raw SDK control metadata reuses the existing equality
decoder and quantum-control signature owner, retaining separate `ctrl2`,
`ctrl` and target ports. It is admitted only above an unmapped raw SDK owner;
named measured controls are not flattened into equal SDK gates. Untagged
ControlledOperation nodes retain the same gate-mapping contract as GateOperation.
Unknown inverse wrappers and undeclared child work remain rejected.

The named selector now distinguishes `ControlledGate(SWAP)` as TwoBitCSwap
from the existing exact-instance CSWAP mapping to CSwap(1), as the original
gate mapper does. Both retain their own source owner. QFT's atomic TwoBitCSwap
reuses the existing 17-instruction exact Fredkin recipe. Named self-adjoint
Hadamard and swap children keep their canonical source direction before control.
No circuit optimization or replacement by an equal-unitary gate is introduced.

| Targets / reversal | Actual positive / negative / two-bit-control primitives | Original/native model availability |
| --- | --- | --- |
| 0 | 0 / 0 / 0 | All seven categories zero |
| 1 | 1 / 21 / 51 | Positive: Clifford 4, rotation 2; other activations separately matched |
| 2, reversed output | 34 / 130 / 246 | Original nested controlled S/S-adjoint model fails |
| 2, no output reversal | 17 / 64 / 138 | Same original model failure |
| 3, reversed output | 65 / 195 / 369 | Same original model failure |

All tested physical streams have no measurements or private work. Each literal
child port, primitive order, operand, parameter and local matrix matches the
unmodified source execution of the explicit adjoint graph. Complete complex
operators, zero work, source-free native resolution, SDK-free append and IR
round trips pass. Zero-target control uses an existing empty Program body on
the caller's control register; it grants no standalone zero-wire IR operation.
All 24 prior complete Programs and bindings match published `9c8061f3f`.

The 60 physical/model receipts retain 24 available original/native model
comparisons and 36 original model refusals. QFT with at least two targets
reaches nested controlled S or S-adjoint, whose original call graph has no
declared decomposition. Native resource mapping remains unavailable for this
nested raw controlled phase instead of pricing its SDK primitive stream.
Forward and inverse model failures are tested explicitly. This is an original
boundary, not a new estimate or a waiver of whole-file parent acceptance.

PauliStringPhasor still needs the named X control and single-qubit Pauli-string
operation wrapper. Other exact named controls, child-private allocations and
previously recorded original failures remain open. No protected API, resource
schema, compiler arithmetic or reference source is changed.

The concentrated batch passes 630 cases without skips in each actual SDK
environment, with 145 local integration cases. Five Python files pass strict
typing, Black/Ruff, scopes, language, links, architecture, IR and API checks.
Counts use the original named primitive basis, retaining CH and local matrix
atoms; they are not a fully synthesized Clifford+T inventory.

### Pauli-string phasor roots and declared control models

Serialized explicit Adjoint/Controlled PauliStringPhasor roots now retain the
single-qubit Pauli-string operation wrapper and the source XGate control
specializations. A single positive control owns CNOT; the exact positive
QBit array `(1, 1)` owns Toffoli and its packed control/target signature.
A packed integer control remains a distinct source predicate. Source
self-adjoint CNOT and X children retain their canonical direction. Temporary
SDK control Qids follow the declared control-before-data order: SDK routines
can sort equivalent controls, so mixed NamedQubit/LineQid ordering previously
changed literal CNOT operands under double positive control. The corrected
order matches the source per-wire stream without optimizing either circuit.

The independent resource adapter now projects a raw child's declared call
body before adding control. It does not control that child's static
single-Clifford or single-rotation estimate. The existing sequence and
predicate model owners perform all arithmetic. Automatic SDK mapping to
ControlledViaAnd remains distinct from an explicitly constructed Controlled
root; tests retain their different costs. Unknown bodies, channels without
source decompositions and unowned nested predicate/inverse origins remain
unavailable. Resource generalization uses this same authoritative mapper.
No Core instruction, public schema or compiler arithmetic changes.

| Pauli word / phase case | Actual positive / negative / two-bit-control primitives | Actual positive measurements / work bits |
| --- | --- | --- |
| Z | 38 / 32 / 88 | 2 / 1 |
| XY | 48 / 107 / 245 | 2 / 1 |
| YZ | 42 / 92 / 204 | 2 / 1 |
| XYZ | 50 / 151 / 317 | 2 / 1 |
| Negative XY coefficient | 48 / 107 / 245 | 2 / 1 |
| XY, both phases zero | 10 / 75 / 157 | 0 / 0 |
| Empty word | 0 / 0 / 0 | 0 / 0 |

The 84 physical/model records cover these cases at four control layouts and
three explicit adjoint depths. Seven independent QEC categories match the
original model. For example, positive XY estimates two Toffolis, two Ands,
four Cliffords, six rotations and two measurements, separately from its
48 actual primitives. Negative XY has no actual measurement or work but its
model estimates two measured And cleanups. Counts retain named atoms and
local matrices; they are not a fully synthesized Clifford+T inventory.

Every immediate child port, per-wire primitive dependency, operand, parameter
and local matrix is compared with the unchanged source. Complete complex
basis columns and every small measurement history verify absolute phase and
clean work, including SDK-free `circuit.append(op)` and IR round trips.
The additional exact positive control pair covers the specialized Toffoli
port, full complex operator and no-work execution. Symbolic XY binds both
phase expressions at nonzero and zero values while retaining the original
two measured erasures; binding does not redispatch a concrete constructor.
Symbolic declared models cover positive, negative and both two-bit patterns.

All 24 prior complete Programs and bindings equal published `11fd9fa27`.
A separately isolated old-selector comparison checks 84 prior adjoint roots:
entry functions and bound CircuitIR JSON are identical. Some unused helper
modules differ because named X controls now resolve to their source owners;
whole-module byte identity is not claimed for those cases.

The empty-word source SDK decomposition returns no operations even for
nonzero phases. The native path follows that source execution boundary and
does not claim a mathematical scalar-phase implementation or standalone
zero-wire IR support. Arbitrary identity-letter words, other unowned named
controls and child-private allocations remain outside this receipt. Original
QFT nested phase model refusals and pending API proposals remain explicit.
Whole-file serialization and resolver parents are not marked complete by
this family closure.

The concentrated regression passes 825 cases without skips in each actual
Cirq 1.7.0/Qiskit 2.5.0 and Cirq 1.6.1/Qiskit 2.0.0 environment. Seven additional
symbolic cases pass in each environment, alongside the three existing ISwap
symbolic cases. Local integration passes 145 cases. Seven Python files pass
strict typing, Black/Ruff, owner scopes, language, links, architecture, IR and
API checks. Locked Qualtran 0.7.0 and the real Kickmix 0.1 Simulator are
asserted before reference execution. The original source is not modified.
