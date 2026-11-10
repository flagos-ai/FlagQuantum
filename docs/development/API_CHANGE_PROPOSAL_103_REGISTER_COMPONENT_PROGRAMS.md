# Proposal 103: Register component programs

## Status and scope

Register-valued graph proposal for G2 of PR #1084. Metadata prerequisites 0.1
through 0.12 are implemented independently. Finite gate-call prerequisite 1.1
implements an additive internal contract 0.6 slice over existing gate SSA.
Packed word prerequisite 1.2 implements internal contract 0.7 for concrete
quantum bit/opaque/unsigned ports and typed rebindings through existing gate
leaves. Source-style mutable construction, automatic work ownership and direct
numerical register traversal remain separate proposals. No owner approval is
declared by this document.

The repository's Core instructions require an integration contract/ADR change
before changing IR. At the initial audit, contract 0.5 admitted gate-level SSA, canonical dynamic
conditions and finite physical profiles. Contract 0.6 adds finite gate calls as
described in prerequisite 1.1. Contract 0.7 adds the concrete word-call and
rebinding dependency slice in prerequisite 1.2.

The user-authorized migration requires shared components to retain the logic of
Qualtran v0.7.0, revision
`bcb17bb25360ed68ce3bc084645db3e929799389`, in FlagQuantum's domain boundaries.
Resource estimates alone cannot implement the required execution graph.

## Existing authority and reproducible gap

Run:

```sh
python -m tools.audit_register_component_scope
```

The checked-in [audit evidence](evidence/register_component_scope_audit.json)
records source fingerprints, dataclass fields, actual verifier diagnostics and
exact static/dynamic round trips. It is a dated capability observation, not a
permanent restriction on future support. Regenerate it after an authorized
boundary change; do not preserve a missing capability to satisfy the audit.

| Existing authority | Reuse | Missing semantics |
| --- | --- | --- |
| `core/ir/program/model.py` | Value identity, blocks, operations, functions | Register type/shape, callee signature and explicit return binding |
| `core/ir/program/verifier.py` | Def-use, linear resources, conditions, diagnostics | Register aliases, per-call ports, type compatibility and call cycles |
| `core/operator_schema.py` | Canonical primitive vocabulary | Variable-width component signatures; a primitive matrix is not a component body |
| `compiler/_component_trace/trace.py` | Issued instruction intervals and child attribution | Executable typed values and connections |
| `compiler/_component_trace/call_graph.py` | Aggregated multiplicity | Instances, output ownership and def-use |
| `compiler/_component_trace/resource_model.py` | Declared costs and sequential peak width | Execution and measurement-branch amplitudes |
| `compiler/_component_trace/width_schedule.py` | Source-derived stable greedy order over declared width connections | Register identity and executable component binding |
| `simulation/classical_*` | Already validated value, codec and phase kernels | A shared traversal that binds graph ports and invokes those kernels |

The audit currently observes `program.opcode_unknown` for a hypothetical call,
kind errors for applying a primitive to an index, and `value.identity` for an
undeclared quantum register kind. None is repaired by registering a matrix gate.
Successful existing static and dynamic round trips remain required regressions.

## Ownership and code placement

| Responsibility | Proposed owning path | Excluded responsibilities |
| --- | --- | --- |
| Immutable register type, port direction, shape and signature | `core/ir/program/registers.py` | SDK types, numerical codecs, graph construction |
| Component instance/callee, typed binding and explicit returns | Existing `core/ir/program/model.py` with narrowly split data records as needed | Callbacks, compilation, execution |
| Register def-use and signature verification | `core/ir/program/register_verifier.py`, invoked by existing verifier | Inferring clean quantum work from SSA alone |
| Bound construction, split/join and finite component expansion | `compiler/registers/component_builder.py` | Numerical simulation, backend selection |
| Gate-stream lowering and instruction interval tracking | Existing Compiler register emitters and `_component_trace` | An additional pass registry or cost engine |
| Numerical graph traversal and value-state updates | `simulation/classical_program.py` | Importing Compiler, upstream libraries or a private graph clone |
| Foreign source differential adapters | `tests/unit/` and reference tooling | Production dependence on Qualtran or Cirq |

These are responsibility assignments, not permission to create all files at
once. A file must have a real accepted consumer when implemented. Core remains
the sole graph authority; Simulator and Compiler consume that same data. There
is no FlagQuantum copy of a foreign package hierarchy or public Bloq/Soquet API.

## Required data contract

The contract-first implementation must specify these details before widening
the production verifier or the current contract checker:

1. **Register type identity:** distinguish opaque bits, unsigned/signed and
   other admitted encodings. Width equality alone is insufficient. Existing
   numerical codec behavior remains in Simulation; Core holds neutral type
   parameters. Concrete and symbolic metadata are distinct from executable
   admission. Do not convert unresolved symbolic widths to integers.
2. **Direction:** inputs, outputs and through ports correspond to the pinned
   source's LEFT, RIGHT and THRU semantics. Names are unique per side, not
   globally. A LEFT word and a RIGHT word with one name may have different types
   and widths. Positional flattening cannot erase this distinction.
3. **Shape:** indices use row-major order; an element's width is separate from
   its array dimensions. Concrete shape with symbolic element width can be
   indexable even though it cannot be physically lowered. Preserve source shape
   and bounds failures at their owning layer.
4. **Zero width:** explicitly declared scalar zero-width registers retain a
   logical port. A shape containing zero has no indexed elements. The source
   `Signature.build` convenience constructor filters zero widths, whereas an
   explicit `Register` does not. Neither physical wire count nor an empty tuple
   is sufficient logical identity.
5. **Connections:** every quantum resource version has one consumer; a call
   consumes current inputs and defines output successors. Classical values may
   fan out. Split/join are typed rebindings, not automatically physical SWAPs.
   Aliases must not let the same resource enter two live call inputs.
6. **Functions:** declare ordered input and output ports and explicit returned
   bindings. Empty returns, missing returns and LEFT-only effects must be
   distinguishable. Calls name finite immutable component definitions; undefined
   callees, inconsistent signatures and cycles are refused. This does not add
   general recursion, closures, dynamic allocation or a classical language.
7. **Work ownership:** allocations, clean/dirty requirements and release are
   explicit. Def-use proves ownership, not quantum zero. Numerical branch tests
   and existing clean-work checks prove actual cleanup. A destructive
   measurement cannot preserve a usable quantum value merely by retaining an
   old reference. Existing nondestructive measure/reset semantics stay intact.
8. **Constants and parameters:** lossless integral table and type parameters
   cannot be squeezed into `Operation.literal`'s floating-point gate constant.
   Their exact immutable representation and symbolic validation must be declared
   before a QROM body is encoded. Parameters are not arbitrary Python callbacks.
9. **Compatibility:** old gate-level records keep their existing identity,
   hashes, verification and canonical-payload round trips. Register bodies need
   real lowering; attaching a payload must not bypass their verification or
   claim a reversible round trip without an exact graph match.

The first executable admission is finite concrete bit/unsigned-register Add and
XOR QROM, using already accepted emitters, numerical kernels and cost models.
This admission is a dependency slice of G2, not a reduction of its final scope.
Other source types remain explicitly unsupported until their matching contracts
and differential tests pass. The final G2 checklist remains authoritative.

## Source call chains and proof obligations

| Pinned source | Native responsibility and evidence |
| --- | --- |
| `_infra/registers.py` Register/Signature | Type/direction/shape identity, per-side lookup, ordering, adjoint and width semantics |
| `_infra/composite_bloq.py` builder and connections | Bound instance identity, consumption, explicit output bindings and split/join |
| `_infra/binst_graph_iterators.py` | Dependency-respecting traversal and deterministic connection grouping |
| `simulation/classical_sim.py` | Input binding, output collection and classical action dispatch; reuse accepted numerical kernels |
| `simulation/classical_sim.py` PhasedClassicalSimState | Preserve original phase, recursive failure and measurement-outcome semantics |
| `bloqs/arithmetic/addition.py` Add | Same primitive stream, carries, feedback, all complex amplitudes and work cleanup |
| `bloqs/data_loading/qrom_base.py` and `qrom.py` | Same XOR table semantics, selection/index order, target shape and emitted source decomposition |

Source witnesses in `test_register_component_scope_audit.py` cover
zero-width ports, opposite-side same-name registers, row-major shapes and
symbolic shape/width separation. They are requirements evidence, **not** proof
of a graph interpreter. The later `test_register_metadata.py` suite compares
the independently implemented standalone Register methods; source Signature
and executable graph behavior remain outside that prerequisite's acceptance.

## Contract-first delivery and acceptance

1. Land the reviewed internal contract delta and one replaceable contract fake
   before domain implementation. Reuse existing Program records instead of a
   parallel graph. Keep fake code test-only, with the same data contract and
   conformance scenarios later consumed by the implementation.
2. Implement Core metadata and verifier in an independent module commit. Compare
   to the unchanged source, including port order, zero-width, side collisions,
   invalid symbolic execution, stale references, missing/extra returns and type
   mismatches. No snapshot regeneration as a repair.
3. Implement Compiler construction and gate lowering, then replace the fake
   without changing its consumer scenarios. Add and XOR QROM must share that
   builder. Reuse the already ported greedy priority and insertion-order policy
   in `width_schedule.py`; do not port another sorter for the same semantics.
   Extending its consumer to executable typed bindings needs separate fidelity
   tests. Enabling instruction attribution must not change emitted IR.
4. Implement Simulation traversal against Core records and existing kernels.
   Compare original graph evaluation, full complex amplitudes, complete small
   measurement histories, side effects on failure and all work outputs. Replace
   a numerical leaf implementation without changing graph consumers.
5. Compare primitive streams, operands and feedback to unchanged source
   decompositions. Report observed primitive counts separately from original
   QEC/resource-model estimates and record source exceptions explicitly.
6. Run existing Program/physical/dynamic, register, architecture, strict typing,
   formatting, scope and public-contract regressions. Run a shortest user
   example through `circuit.append(op)` for both concrete consumers.

Each accepted implementation updates the item checklist and overview. This
proposal and its audit do not check off any source module as migrated. G3,
automatic merge, CI waiting, public root exports, public CircuitIR serialization,
`IR_VERSION`, Runtime result schemas and execution defaults are outside scope.

## Metadata prerequisite 0.1

This separately implementable prerequisite belongs to the user-authorized G2
shared-register work. It adds no Program IR opcode, value kind, model field,
public API, serialized schema or executable profile. The version 0.5 Program
contract remains authoritative and unchanged. The broader call/return proposal
above remains draft.

The shared symbolic-object predicate moves from Simulation to
`core/value_parameters.py` unchanged. Its current internal consumers move with
it; the old private module is removed rather than retained as a duplicate owner.
Core metadata and numerical consumers must use this one predicate. Optional
Sympy handling, raw checker-result propagation and import-failure behavior stay
unchanged. Concrete native operation must not acquire a Sympy dependency.

`core/ir/program/registers.py` admits immutable, data-only register metadata:

- `RegisterType(encoding, quantum_bits, classical_bits=0, parameters=())`
  identifies a prevalidated encoding with exact quantum/classical element
  widths and ordered immutable type parameters. Different encodings are not
  interchangeable merely because their widths agree. This descriptor does not
  implement upstream dtype constructors or numerical bit encoding. Construction
  of a valid descriptor is not certification that a type can be executed.
- `RegisterSide` retains input, output and through flag semantics.
- `Register(name, dtype, shape=(), side=THRU)` retains the source's shape
  conversion, scalar/array distinction, symbolic shape refusal, row-major
  indices, total quantum/classical widths and exact adjoint-side behavior.
  It validates a native nominal `RegisterType`; foreign SDK type objects do not
  enter Core. Width validity and encoding-specific constraints belong to the
  later type admission/binding boundary, not a repaired source Register check.

The first contract fake supplies Add and XOR QROM signatures as ordered tuples
of these same immutable register records. It is test-only; it must not invent a
second executable graph or pretend a Signature class, graph builder, function
return or numerical traversal is complete. Production register metadata must
replace its metadata behavior while the same source-differential scenarios
remain unchanged. The source Register/Signature distinction and pending broader
metadata methods remain explicit checklist boundaries.

Acceptance requires unchanged source Register comparisons for opaque, bit,
unsigned, classical and shaped metadata; all side/zero-width/symbolic edges;
two real Add/QROM signatures; unchanged numerical predicate consumers; and
existing Program IR/public-contract regressions. The result makes the typed
graph possible but does not itself enable it. Tests report metadata agreement
separately from already accepted quantum amplitude and cost evidence.

## Signature prerequisite 0.2

This next standalone G2 metadata slice adds `Signature` in
`core/ir/program/signatures.py`, using the existing native Register records.
It leaves the version 0.5 Program contract and public API unchanged; it does not
authorize or implement new component-call opcodes or a graph interpreter.

The exact ordered product, sum and minimum/maximum helpers from pinned
`symbolics/math_funcs.py` belong in `core/value_math.py`. Register metadata and
Compiler cost arithmetic consume that same implementation. The existing Compiler
resource-math module keeps its rounding/logarithm responsibility and exposes the
shared extrema through its existing imports, without a second implementation.
Plain built-in numbers retain the native no-Sympy fast path; integer subclasses
and other symbolic-object protocols must not bypass source classification.
Empty-pack and empty-iterable diagnostics follow the source separately.

`Signature` retains per-side name uniqueness, LEFT/RIGHT namespace separation,
ordered grouping, cached through-only classification, positional/slice access,
hashing, adjoint and independently computed quantum/classical/combined widths.
Its width convenience builder follows source filter, one-bit selection and
opaque-width validation order. The dtype convenience builder intentionally
filters on quantum width, including the source's omission of classical-only
ports. Explicit construction can retain those ports and zero-width registers.
This is metadata behavior, not execution admission for arbitrary encodings.

Acceptance compares these methods and failure order with the unchanged source,
replaces source-backed Add/QROM signature consumers with the native Signature,
and reruns existing primitive/amplitude/branch/work/resource regressions. A
separate contract-first change is still required for typed connections, explicit
function results and executable component calls.

## Audit validation

The initial audit has five tests, including three pinned-source witnesses. With
the existing internal-level conformance suite, 36 tests pass without skips or
warnings. The audit tool passes strict typing, Black and Ruff; architecture and
the existing multi-level contract gate pass. Protected source and contract files
are fingerprinted but unchanged. This evidence closes the scope investigation,
not the contract fake, graph implementation, native metadata port or G2.

## Type constructor prerequisite 0.3

This G2 slice adds `core/ir/program/register_types.py` as the owner of native
factories for bit, opaque, unsigned and bounded unsigned quantum metadata.
Factories return the existing immutable `RegisterType`. They preserve the
construction behavior of pinned QBit, QAny, QUInt and BQUInt without introducing
foreign objects, bit codecs or new Program IR fields.

The internal interfaces are `quantum_bit_type()`,
`opaque_register_type(width)`, `unsigned_register_type(width)` and
`bounded_unsigned_register_type(width, bound=...)`. An omitted bound is distinct
from explicit `None`. Default exponentiation precedes symbolic classification.
Opaque construction classifies symbolism before checking Python integer
membership. Unsigned construction performs no eager width validation. Bounded
construction retains the original variadic symbolic short circuit and the
strict upper-capacity comparison. Negative, zero, bool and symbolic widths keep
the original constructor outcomes. Native error text replaces the foreign type
name, while preserving failure kind and evaluation order.

Signature's width builder consumes bit/opaque factories after its existing
truth filter and one-bit selection. QROM consumes opaque control/target and
bounded selection factories after its existing dimensional and axis-fit
validation. Its existing fit-axis diagnostic and physical binding checks remain
at that boundary. Ports retain big-endian word order, row-major target shape,
side, width and bound identity. These factories allocate no workspace and emit
no instructions. Add resource and QROM consumers must preserve their existing
primitive, amplitude, measured-branch, workspace and cost regressions.

The test-only source-backed factory stand-in returns the same native descriptor
type. Its constructor scenarios must run unchanged when native factories replace
it. Differential cases include omitted/explicit bounds, symbolic literals,
Python/NumPy scalar distinctions and original failure ordering. Numerical codecs
remain Simulation-owned. Type compatibility between graph ports, other dtype
families and executable typed graph admission require subsequent slices.

The prerequisite contract stand-in passes 124 source-backed scenarios before
native implementation. It reads declared width directly, avoiding the separate
lazy `_bit_encoding` construction and its additional numeric validation. This
records constructor behavior independently from later codec/property behavior.

## Integer and classical type prerequisite 0.4

Extend the existing internal constructor module with signed two's-complement and
ones'-complement quantum words and classical bit, unsigned, signed,
ones'-complement and bounded unsigned words. The source is the unchanged QInt,
QIntOnesComp, CBit, CUInt, CInt, CIntOnesComp and BCUInt constructor call chains.
Two's-complement and classical unsigned constructors retain their lack of eager
width validation. Ones'-complement constructors reject only a Python integer
one, including True. NumPy/Sympy integer one follows the source's separate path.

Native signed encoding remains `signed`; ones' complement uses
`ones_complement`. Classical integer factories store zero quantum width, the
original classical width and `("storage", "classical")` as their first type
parameter. Bounded classical words append `("bound", bound)`. The marker keeps
zero-width classical and quantum integers nominally distinct even when both
physical width counts are zero. CBit uses the existing `RegisterType("bit", 0,
1)` identity. Existing directly constructed descriptors retain their values.
Source-backed tests map these identities explicitly, including empty scalar
ports and Signature's deliberate classical-only convenience-filter omission.

BCUInt computes its omitted bound before classification and retains its `bound`
diagnostic wording. The shared bounded resolver preserves the existing BQUInt
construction order and diagnostic wording. The existing ones'-complement width
guard moves unchanged from Simulation to Core. Numerical codecs and signed
truncation consume that same guard at their existing call sites, with no second
validation authority and no changed validation timing.

The source-backed constructor fake and native factories must run the same new
scenarios. Compare declared widths without invoking lazy codecs, preserve raw
parameter types and constructor failures, test quantum/classical zero-width
identity and Signature counts, and feed constructed widths into the existing
numerical codecs. Rerun gate-stream, complex-amplitude, all small measurement
branches, zero-workspace and resource regressions for the existing signed
conversion and Add/QROM paths. Add's existing symbolic cost consumer switches
its direct unsigned descriptors to the shared unsigned factory.

This standalone metadata slice retains Program IR 0.5 and current public API
and serialized schema contracts. Montgomery, fixed-point and finite-field
constructors, graph-port type compatibility and executable typed connections
remain later G2 dependencies.

The source-backed prerequisite adds 182 constructor and consumer scenarios.
Together with the prior constructor slice, 438 tests pass before new native
implementation. Source width fields are read directly to keep lazy numerical
codec validation outside constructor evidence.

## Montgomery type prerequisite 0.5

The existing Core factory module gains `montgomery_register_type(width,
modulus=None)` and `classical_montgomery_register_type(width, modulus=None)`.
They adapt the unchanged QMontgomeryUInt and CMontgomeryUInt constructors to
existing RegisterType records. Encoding is `montgomery`. Classical storage keeps
its existing first `storage` parameter. A supplied non-None modulus follows as
`("modulus", modulus)`. An absent or explicit None modulus adds no parameter,
preserving their source equality. Width and modulus retain their declared
values and scalar types. Construction performs no arithmetic, coercion,
truth-value check, primality check or lazy numerical codec validation.

Montgomery RegisterType symbolic classification follows the original inherited
QCDType method, which examines width and ignores modulus. This encoding-specific
rule belongs in the existing RegisterType method. The generic parameter-aware
rule remains authoritative for other encodings. This distinction includes a
concrete width paired with a symbolic modulus. It must have a pinned-source
witness before use in graph compatibility.

The source-backed constructor stand-in and native factories share raw field,
zero-width identity, default/None and declared scalar-type scenarios. After
implementation, compare source symbolic classification and shaped Signature
widths, replace source-backed descriptors in existing numerical Montgomery
consumers, and exhaust small conversion/product/inverse domains. Keep source
numeric cast asymmetry and source composite-modulus algorithm limitations.
Run the existing Montgomery primitive-stream, exact-adjoint, complex-amplitude,
measurement-branch, workspace and resource regressions separately. Extend the
existing executable Montgomery example to consume the metadata widths while
retaining its `circuit.append(operation)` path and distinct model/emitted counts.

This prerequisite preserves the Program IR 0.5 and public serialization/API
contracts. Constructor metadata adds no quantum gates or allocated workspace.
Numerical arithmetic remains Simulation-owned. Fixed-point/finite-field types,
full graph-port compatibility and typed component execution remain later slices.

Before native implementation, all 326 source-backed constructor, identity and
Signature consumer scenarios pass. They read declared source fields directly.

## Fixed-point type prerequisite 0.6

Core's existing register type module gains `fixed_point_register_type(width,
fractional_width, signed=False)` and
`classical_fixed_point_register_type(width, fractional_width, signed=False)`.
They return existing RegisterType records with encoding `fixed_point`, ordered
`fractional_width` and `signed` parameters and original declared scalar values.
Classical storage retains its first storage marker, including zero-width words.
Both adapt the unchanged QFxp/CFxp eager shape checks and diagnostic order.
Only the diagnostic type name QFxp changes to the native fixed-point register.
The inherited quantum wording `num_qubits` remains for both source counterparts.

The shape decision tree belongs in Core as `validate_fixed_point_parameters`.
Its default diagnostic context retains the original _Fxp numerical codec text.
The two type constructors select the source QFxp/CFxp context. Simulation's
existing `validate_fixed_point_shape` keeps its signature and validation sites
and delegates to this shared guard. Width classification remains repeated at
the original points, including its short circuits and observable protocol calls.
No eager integer-domain, encoding, real quantization or positivity check is added.

Fixed-point symbolic classification examines width and fractional width, as the
source does. It ignores signed and storage parameters. This encoding-specific
rule extends the existing RegisterType method. Generic parameter classification
for other encodings and the Montgomery width-only rule remain unchanged.

Acceptance uses a source-backed stand-in before implementation and the same
native constructor scenarios afterward. Compare raw fields and scalar types,
eager failures, zero-width storage identity, symbolic/NumPy inputs, protocol
call order and shaped Signature consumers. Reuse constructed widths/parameters
in existing fixed-point numerical codecs and exhaust small signed/unsigned
stored integer domains. Compare shared guard diagnostic contexts and replace
the guard in its real Simulation consumer. Rerun existing phase/fixed-point
primitive-stream, exact-adjoint, complex-amplitude, measured-branch, workspace
and separately counted resource tests, plus the existing append example.

The slice preserves current Program IR and public API/schema contracts. It
introduces no real quantization dependency, execution opcode or foreign type.
Finite-field constructors and typed graph connections/calls remain subsequent
G2 work.

The source-backed prerequisite passes 430 constructor/Signature/protocol cases
before native implementation, including the eager failure contexts.

## Shared width arithmetic prerequisite 0.7

Finite-field metadata requires the source scalar width calculation. The existing
Compiler `ceil`, `floor`, `log2` and `bit_length` helpers move to Core's existing
`value_math.py` owner and remain available through their former Compiler imports.
The original symbolics.math_funcs decision tree determines their outcomes.
Concrete ceil/floor use int(np.ceil/floor(value)), retaining scalar conversion
and diagnostics. Symbolic cases retain the original Sympy functions. Bit length
retains its original symbolic-first zero interval and floor/log2/ceil chain.
NumPy is already a required production dependency. Scalar NumPy results stop at
the int conversion and do not become Core contract types or execution kernels.
No new dependency, dispatcher, width IR or numerical callback interface is added.

The original native no-Sympy behavior for concrete built-in values remains
supported. Existing single-object symbolic protocols, integer subclasses and
NumPy scalars must follow the source classification, including return types.
Source and native cases compare values, result classes, warnings and exact
exceptions. Protocol traces check short circuits and evaluation order.

A source-backed stand-in must pass before Core implementation. Compiler's
existing import bindings then reuse the same authoritative functions. Replace
the width dependency in its actual Hamming/QROM/resource consumers and rerun
source primitive streams, complex amplitudes, measured branches, zero work and
separate model/emitted resource accounting. This slice supplies the next
finite-field metadata dependency and leaves typed graph admission unchanged.

The prerequisite stand-in passes 109 source-backed cases before implementation.
The retained audit records NumPy scalar return-type differences in the former
Compiler helpers. These differences motivate preserving source classification
and concrete scalar rounding at the shared owner.

## Finite-field numerical encoding prerequisite 0.8

Simulation owns `FiniteFieldEncoding`, adapting the unchanged `_GF` numerical
codec and shared QGF/CGF polynomial construction chain. This is a numerical
encoding object with cached field arithmetic, not a second register type system.
Core RegisterType remains the authoritative neutral metadata record. Converting
this numerical object to a register descriptor is a subsequent explicit boundary
because source construction accepts inputs whose width fails only on access.

The constructor retains characteristic and degree without coercion. An omitted
polynomial invokes the original symbolic test and concrete GF default selection.
Explicit None bypasses default selection. Both paths then import galois for the
original converter, retaining Poly objects by identity and converting other
inputs using Poly.Degrees. Field order, bit width and field class are cached only
on access. The source degree > 1 condition and GF arguments, including
verify=False and python-calculate, remain exact. Numerical scalar validation
requires the actual field class. Array validation remains range-only. Scalar
and array bit encoding use the existing big-endian integer and default vectorize
boundaries, including their existing native diagnostic terminology.

Galois 0.4.11 is the locked reference's MIT-licensed numerical dependency. It
remains an explicitly installed optional Simulation dependency. This slice adds
no package installation, manifest requirement, root export or automatic fetch.
Imports are lazy until construction, including construction with explicit None.
Galois owns field arithmetic, polynomial objects and FieldArray values exclusively
inside Simulation. The finite-field codec is the replacement boundary and can
be compared directly with the source codec without changing its consumers.
A future replacement must preserve polynomial identity/conversion and the tested
failure timing. Core must never import galois or retain its objects.

Acceptance first runs source-backed construction and lazy-property cases before
implementation, then the same cases against the native codec. Cover symbolic,
NumPy and invalid raw inputs, omitted versus None, explicit polynomial identity,
prime/extension fields, scalar and array round trips and validation failures.
Replace source numerical bindings in actual binary-field multiplication consumers
and compare their complete primitive streams. Rerun existing exact complex-state,
measurement-branch, workspace-cleanup and separate model/emitted-count tests.
Typed graph ports and neutral finite-field descriptor projection remain open.

The source-backed prerequisite passes 157 cases before native implementation.
Composite-characteristic cases can raise RuntimeError in the locked galois
lookup-table builder. Preserve the observed exception at its original stage.

## Finite-field neutral projection prerequisite 0.9

Simulation's existing FiniteFieldEncoding gains an explicit
`register_type(*, classical=False)` projection into Core's existing RegisterType.
This is the explicit width-resolution boundary. The numerical constructor and
its cached properties keep their current timing. Projection reads bitsize,
retains characteristic, degree and cached order as named scalar parameters and
uses encoding `finite_field`. Classical storage retains its first storage
marker and puts width in classical_bits. Quantum width uses quantum_bits.
No new public type, serialization schema, Core numerical dependency or parallel
register hierarchy is introduced.

An explicit None polynomial contributes no polynomial parameters. A present
polynomial contributes its exact integer representation plus its coefficient
field's characteristic, degree, primitive element and irreducible polynomial,
all converted to Python integers inside Simulation. These values distinguish
the canonical field classes created by the locked galois factory. In particular,
polynomials with identical integer representation over different coefficient
fields remain distinct. Prime-field degree one retains a supplied polynomial
in nominal metadata even though numerical gf_type ignores it. The projection
must not resolve the target gf_type or normalize None to a default polynomial.
Manual galois FieldArray subclasses bypassing its canonical field factories
remain outside this projection's nominal-identity acceptance.

Core's existing RegisterType symbolic method recognizes `finite_field` and
examines the characteristic and cached order parameters, in source order.
It ignores physical widths, degree and polynomial parameters, preserving QGF
and CGF classification after explicit width resolution. Projection failures are
explicit projection failures. They must not be described as eager source
constructor validation. Consumers of unresolved raw finite-field declarations
still use the existing numerical constructor until they require a descriptor.

A source-backed projection stand-in must run before implementation. Compare
QGF/CGF widths, symbolic classification, shaped Signature totals and pairwise
nominal equality for canonical fields. Include omitted versus None, symbolic
and NumPy inputs, raw-field errors at projection, different polynomials,
coefficient fields and primitive elements. Verify that only neutral scalars
cross the boundary and target field construction is not invoked. Feed both
source/native descriptors into the actual append multiplier consumer, compare
the unmodified quadratic decomposition, full complex amplitudes and independent
resource categories. Rerun existing measured erasure and clean-workspace tests.

The source equality audit found that comparing a None polynomial with a present
Poly raises TypeError inside galois Poly.Like. Neutral RegisterType record
equality distinguishes those snapshots without replaying numerical-library
operators. Both-None and both-present canonical polynomial comparisons retain
source nominal equality. The cross-None source exception is recorded separately
and must remain visible when implementing the later source dtype-consistency
operation. This prerequisite does not claim full source equality semantics.

Before native implementation, 132 source-backed projection and Signature cases
pass, including the separately recorded cross-None equality exception.

## Register type consistency prerequisite 0.10

Core's internal `register_compatibility.py` adapts the unchanged
check_dtypes_consistent decision tree over existing RegisterType descriptors.
RegisterTypeCheckingSeverity retains the source Enum names and values LOOSE=0,
ANY=1 and STRICT=2. The default remains LOOSE. This standalone consistency
predicate does not authorize executable type conversion or alter Program IR.

Evaluate nominal equality and quantum-width equality before severity decisions.
Equal declared types and equal one-qubit types are admitted first. STRICT then
refuses remaining cases. Quantum opaque registers admit matching quantum widths
before the ANY refusal. LOOSE permits quantum signed/unsigned/bounded/Montgomery
integer interchange, and unsigned/bounded/Montgomery/finite-field compatibility
with wholly integral or wholly fractional unsigned fixed point. Preserve the
source exclusion of finite fields from the integer interchange group and of
ones' complement and classical encodings from numeric conversion groups. Source
quantum-width-only opaque comparisons, including classical operands, remain
observable. Classical bit counts are not an added admission criterion.

Finite-field nominal comparison replays the source None/Poly TypeError when
storage, characteristic and degree coincide and exactly one polynomial is
present. Other canonical descriptor equality remains Core record equality.
The replay precedes dimensional and severity checks, as the source equality
does. It introduces no galois import or external object in Core. The source
exception text is retained as diagnostic data. Fixed-point num_int evaluation
reuses the existing shared shape guard at the original lazy access point.

A source-backed stand-in runs the complete supported descriptor cross-product
under all three severities before implementation. Then the same cases run
through native consistency. Compare values, scalar result classes and errors,
including symbolic widths, quantum/classical mix, one-bit precedence, opaque
precedence over ANY, signed fractional exclusion, field/integer asymmetry and
cross-None exceptions. Demonstrate consumers substituting source/native checks
while retaining existing Signature and appended gate bindings. Existing exact
primitive-stream, complex-state, measured-branch, zero-work and independent
resource regressions remain required. Typed graph binding and execution remain
the next dependency after this predicate.

Before implementation, all 35,643 source-backed decisions pass across 109
supported descriptor declarations and the three severity settings.

## Indexed binding prerequisite 0.11

Core owns `RegisterBinding(value, register, index=())` in
`program/register_bindings.py`. It associates an existing `ValueRef` identity
with one logical register element. The index converter and bounds checks follow
the pinned Soquet constructor. A Python integer becomes a singleton tuple.
Rank is checked before upper bounds. Negative indices retain the source
constructor behavior. Concrete shape and symbolic element width remain separate.
A scalar zero-width element retains an identity. An empty array has no elements.
This metadata does not assign register semantics to an existing value kind.
Admission of register-valued operations still requires the Program contract delta.

Compiler owns `process_register_bindings(registers, bindings, context, visit)`
in `registers/component_bindings.py`, adapting `_process_soquets` without
introducing a second graph. Core contains no array conversion. Compiler applies
the source NumPy conversion, name bookkeeping, row-major indexing and nominal
binding assertion. The callback runs before the shared LOOSE dtype check, so a
callback failure precedes a dtype failure and already recorded callback effects
remain visible. Missing names fail when encountered. Surplus names fail after
all declared ports. Oversized arrays, scalar-like singleton arrays, zero-element
arrays and partial failure follow the original indexing behavior. This helper
does not add an exact-shape check that the source does not perform.

The caller owns quantum single consumption and classical fanout. This traversal
neither allocates nor consumes resources by itself. The later component builder
must reuse it for both calls and explicit returns, operating on the existing
Program records. Split/join, allocation, return completeness and quantum cleanup
remain separate obligations. This prerequisite adds no opcode, model field,
serialized schema, public export, numerical execution or physical wire mapping.

The contract fake runs the unchanged source constructor and binding processor.
The same scenarios then consume the native implementations. Cover indices,
symbolic shape versus width, scalar/array and empty ports, malformed arrays,
missing/extra names, dtype compatibility, callback ordering and failure effects.
Use actual Add and XOR QROM signatures and initial bindings from
BloqBuilder.from_signature. Check callback traces independently of gate/resource
regressions. Metadata traversal emits zero physical operations. Existing append
consumers supply the separate primitive, complex-amplitude, measurement-branch,
clean-workspace and resource evidence.

## Binding collection prerequisite 0.12

Compiler's existing `component_bindings.py` gains
`create_register_bindings(value, register, available=None)`, following the
pinned `_reg_to_soq` branches. It groups indexed Core RegisterBinding metadata
around the supplied existing value identity. A scalar returns one binding.
A shaped register returns an object ndarray with the exact declared shape and
row-major elements. A zero-width scalar remains present, whereas a zero-length
array has no elements. Symbolic shape refusal precedes array allocation.
Symbolic element width alone remains valid metadata. This grouping does not
allocate an SSA value or quantum workspace. Callers remain responsible for value
definitions and any eventual executable admission.

When supplied, the availability set receives each newly constructed binding
immediately. Earlier additions survive a later registration error. Omitting the
set simply omits that bookkeeping. Creation performs no direction filtering or
dtype conversion. Those decisions belong to the caller's Signature traversal.
Core retains indexed data ownership and Compiler retains NumPy object layout.
The same binding processor from prerequisite 0.11 consumes these collections.

Run source-backed creation scenarios before implementation, then substitute the
native function without changing consumers. Cover scalar/shaped/empty and
symbolic metadata, exact array shape and dtype, optional availability, callback
failure effects during set registration, all directions and creation-to-binding
round trips. Reuse the Add and XOR QROM append/branch/resource acceptance from
0.11 with both manually supplied and factory-created binding collections.
Finite component calls, explicit returns and general graph traversal remain the
next contract slice.

## Call and explicit-return contract slice 1.0

This is the proposed structural delta for the existing Program authority. The
source-backed fake in `tests/unit/test_register_component_calls.py` executes the
unchanged pinned `BloqBuilder` through `from_signature`, `add_d`/`add_t`/`add`,
`_add_binst`, `_add_cxn` and strict finalization. The scenarios establish builder
requirements before native implementation. They currently run against that
fake alone. At this slice's initial contract commit, version 0.5 remained the
executable authority. Prerequisite 1.1 below subsequently adds a narrower
finite gate-call implementation under contract 0.6. No owner approval is
asserted by this proposal or by passing source witnesses.

### Proposed storage in existing records

| Existing record | Additive field or vocabulary | Default and meaning |
| --- | --- | --- |
| `ValueRef` | Add `register` to the internal kind vocabulary and linear kinds | One owned element token, including a zero-width scalar. Its type comes from its defining function input or call output port. Existing kinds keep their semantics. |
| `Operation` | `callee: str \| None = None` | Present only for opcode `call`. The name resolves to one finite definition in the enclosing `ProgramRecord`. Operands and results are the sole input/output value lists. |
| `BlockRecord` | `return_values: tuple[ValueRef, ...] \| None = None` | `None` retains the existing gate-only implicit return. `()` is an explicit empty return. Register functions require an explicit return. |
| `FunctionRecord` | `registers: tuple[Register, ...] \| None = None` | `None` selects the existing gate-level function rules. A tuple declares ordered register ports, including an explicitly empty signature. Reconstruct existing `Signature` for per-side validation. |
| `ProgramRecord` | No additional graph or payload field | Existing functions contain definitions and existing operations contain calls. |

Use an immutable tuple of existing Register descriptors in the function record.
The cached, mutable Signature object is a validation/view helper. Adding a second
signature class or serializing a foreign graph is unnecessary. Calls refer to
callee names without copying signatures. A verifier resolves the callee and
checks both positional sequences against the resolved signature.

For each side, flatten `Signature.lefts()` or `rights()` in declaration order,
then each port's `all_idxs()` in row-major order. One element contributes one
ValueRef even at width zero. A dimension of zero contributes no ValueRefs but
its port declaration stays in the signature. Two array elements always receive
different `(scope, index)` identities. The existing metadata collection factory
can group indices around one ValueRef, so its output cannot be used directly as
an executable element-identity allocator. The builder must issue distinct
existing ValueRefs for every element, while reusing its array and index rules.

A call consumes operands in the callee's left-port order and defines fresh
results in its right-port order. An explicit return consumes its listed values
in the enclosing function's right-port order. Input and output names retain
separate namespaces. A LEFT word and RIGHT word with the same name can have
different widths. Bindings resolve by side before flattening. Type compatibility
uses the already accepted LOOSE predicate after the builder's consumption
callback, matching the source's failure order.

### Ownership and verification

The first structural slice admits finite single-block register functions.
Their entry arguments are the flattened left ports and their explicit return
values match flattened right ports. The verifier must reject unknown or repeated
function names, missing callees, repeated definitions, undefined values,
wrong arity or dtype, stale and aliased inputs, missing explicit returns,
unused tokens and call cycles. The cycle check includes unreachable definitions.
Calls with conditions, parameters, literals or physical data require separate
admission and fail closed in this slice. Multiple blocks and cross-profile
calls also require explicit support before execution.

Every `register` token has exactly one consumer, including a classical-register
handle. The pinned builder's `_available.remove()` applies to CBit as well as
quantum words. The source witness records that behavior. Existing `bool` SSA
values retain fanout for dynamic measurement feedback. This narrows item 5 in
the initial data-contract discussion, which allowed classical fanout without
distinguishing SSA booleans from owned register handles. Numerically copying a
classical value is a separate explicit operation and cannot excuse reusing a
consumed source handle.

Failed source calls reserve an instance index and insert the instance before
binding inputs. Earlier consumed values and recorded connections survive a
later missing name, dtype mismatch, alias, short array or surplus name. Failed
returns likewise retain the consumed prefix. Retrying the same return then
fails on its first already consumed input. The mutable native builder must make
those effects observable through the same contract observations. Failed
construction state is kept separate from an admitted immutable ProgramRecord.
A verifier must inspect the actual definitions, operands and returns rather
than trust successful callbacks or an attached canonical payload.

Source `final_soqs()` returns `{}` when no RightDangle appears in the graph,
even with a declared empty-shaped output port. Preserve the declared signature
and empty returned-value tuple in native records. Reconstruction of an empty
array from the signature is a native consumer decision that must be reported
separately from source helper output equality. The test records this source
boundary explicitly.

### Compatibility and executable follow-through

Append optional fields after existing dataclass fields. With all new fields at
their defaults, preserve existing equality, hash tuples, verifier diagnostics,
physical-profile dispatch and exact static/dynamic payload round trips. New
non-default fields participate in identity. Gate-only records carrying callee
or register-return metadata must fail verification instead of being silently
restored from `canonical_payload`. Public CircuitIR schemas and root APIs keep
their current contract throughout this internal change.

This structural slice introduces no generic component-parameter bag. Exact
integer tables and instantiated algorithm parameters remain with their existing
Compiler-owned immutable component objects while constructing concrete function
bodies. They must never be converted to `Operation.literal` merely to retain a
reference. Before persisting leaf data for numerical dispatch, specify an exact
neutral representation in a separate contract delta. An unresolved leaf is
refused for execution. Resource models alone cannot supply a function body.

The next delivery replaces the source fake with a Compiler builder that
produces these Core records, with Core verification as an independent commit.
The existing Add and XOR QROM consumers then exercise real gate lowering and
Simulation traversal. Split/join, allocation and clean/dirty release need their
own typed operations and explicit ownership rules before those consumers can
claim graph execution. Their existing emitter tests remain required regression
evidence. Passing the call fake alone certifies neither new primitive counts nor
complex amplitudes, measurement branches or workspace cleanup.

## Finite gate-call prerequisite 1.1

The first native implementation stages the proposed storage by adding only
`Operation.callee` and `BlockRecord.return_values`. It keeps existing value
kinds, FunctionRecord fields and all public schemas. Concrete existing Add and
XOR QROM emitters already produce gate-level bodies with explicit workspace
wires. Those bodies supply executable leaves before packed register values,
split/join and allocation/release are admitted. This prerequisite establishes
finite calls over existing typed gate SSA. Register-valued calls remain in the
larger slice 1.0 and retain its separate acceptance requirements.

Internal contract 0.6 admits single-block functions with explicit returns for
this profile. A call names a function in its ProgramRecord, binds its entry
arguments positionally and defines successors matching its explicit returned
value kinds. Callee names are unique and nonempty. Every definition, including
unreachable definitions, is checked. Missing callees, cycles, absent explicit
returns, non-return terminators and mixed physical profiles are refused.
Conditions, literal constants and parameter bindings attach to the leaf's
existing primitive operations, not to the call itself. A callee field on a
primitive opcode is invalid. Calls add no implicit workspace allocation.

Explicit return tuples participate in def-use and linearity. Every returned
value must be defined, each qubit must be consumed exactly once, and omitted
live qubits are errors. Empty and missing return tuples are distinct. Existing
implicit gate-only returns keep their old behavior. A call result's type comes
from the resolved callee's explicit returns. Classical bool and scalar values
retain the existing fanout rules. Quantum output positions can be reordered,
with the order carried by the explicit returns until lowering resolves them.

Compiler owns `inline_program_calls` in `compiler/registers/program_calls.py`.
It validates the complete module, binds one SSA environment per call, expands
finite calls into the existing primitive operations and remaps conditions and
parameters through the same environment. Its output is a single-function
ProgramRecord with explicit returns. Source order within a function is retained.
The inliner uses an explicit stack so valid deep finite call chains do not rely
on the Python recursion limit. It validates its output and emits no numerical
kernel or cost estimates. Consumers can replace a flat body with a hierarchical
body without changing the emitted public circuit or append execution.

Core's attested `program_to_circuit` still verifies and independently re-lowers
the canonical payload. For a flat body with explicit returns, its expected
returns are the final successor of each original physical input wire, in wire
order. A permuted or additional return cannot disappear during restoration.
Hierarchical bodies require Compiler expansion before restoration. A canonical
payload cannot hide a malformed body, an extra definition or a changed return.
Legacy records with absent new fields retain their old hashes and equality.

Acceptance includes nested calls, classical output and parameter/feedback
remapping, duplicate or unknown callees, unreachable cycles, missing/extra or
stale returns, redefined identities, mixed profiles, deep finite chains and
payload tampering. Actual Add and QROM leaves must survive native hierarchy
expansion with their existing source primitive dependencies, operands and
feedback, full complex measurement-branch amplitudes and clean workspace.
Their measured primitive counts and independent upstream cost models continue
as separate observations. No packed-register graph claim follows from this
bit-level prerequisite.

The inliner re-canonicalizes DNF after boolean argument substitution. Reversed
argument order can reverse value identities, and aliasing can merge identical
conditions or make a conjunction contradictory. Duplicate conditions and clauses
are collapsed. Contradictory clauses are omitted. If every clause is false, the
conditional gate forwards its quantum operands to its results and emits no gate.
This is exact specialization under the native call binding, recorded separately
from unchanged upstream decomposition parity. It requires no new boolean opcode.


## Packed register and gate rebindings prerequisite 1.2

This prerequisite fixed the representation in `cd1a4e05` before independent
implementation. The machine contract delta `2e1545d3`, Core `d1d296f9` and
Compiler `b2877bee` now implement its concrete word profile under contract 0.7.
Source-backed witnesses in
`test_register_rebinding_contract.py` exercise the unchanged builder, including
zero-width scalar ownership and failed join consumption. They do not implement
native register execution. Existing numerical codecs and physical-wire
split/join helpers remain authoritative in their own domains.

### Exact storage and type derivation

Append `FunctionRecord.registers: tuple[Register, ...] | None = None` and
`Operation.register_type: RegisterType | None = None` after existing fields.
Default fields preserve the existing hash tuple and physical JSON schemas.
`register_type` is present only on `join_register`. Its exact immutable neutral
descriptor states the destination word type, including width zero. A split
infers its word descriptor from its operand's defining port or join. No mutable
type environment, foreign dtype or table payload is persisted beside the graph.

Add the internal linear value kind `register`. A value identifies one owned
logical element, even at width zero. Function arguments bind left ports and
returns bind right ports in declaration order, then each port's row-major
`all_idxs()` order. Empty-shaped arrays contribute no values. Each element has
a distinct `(scope, index)`. The existing collection helper groups metadata
around a supplied identity and must not allocate executable identities.

The first executable descriptor set is concrete quantum `bit`, `opaque` and
`uint`, with built-in integer nonnegative width, zero classical width and no
extra parameters. `bit` has width one. Metadata constructors retain their wider
symbolic and negative-width behavior. Executable admission rejects unresolved,
negative, classical or unsupported descriptors before lowering. This bounded
admission does not silently reinterpret other encodings.

A register function has one explicit-return block. Every declared argument and
return has kind `register`. Its body may also contain gate-level SSA values.
A gate function retains `registers=None` and cannot accept or return a register
value. Calls bind the resolved entry/return kinds and, for register values, the
producer's inferred descriptor against the callee's declared descriptors through
the existing LOOSE predicate. Register-type flow comes from function inputs,
resolved call outputs and join target descriptors. Returned bindings are checked
against right ports through that same predicate. Direct cross-profile operands
are refused. A register function may call a gate leaf after explicitly splitting
its words into qubits and rejoin those qubit successors before returning words.

### Rebinding operations

`split_register` consumes exactly one `register` and defines one ordered `qubit`
per bit of its inferred descriptor. `join_register` consumes the declared
number of ordered qubits and defines exactly one `register` with its stated
descriptor. These primitive qubit SSA values are the existing gate values, so
no second logical-bit kind or bridge graph is introduced. Source QBit register
handles in the eventual source-like builder still require the builder's explicit
binding policy. They must not be confused with untyped numerical booleans.

Both rebindings forbid conditions, literals, primitive parameter bindings,
physical data and callees. A split forbids `register_type` because its producer
already states that type. A join requires it, including zero width. Type and
arity inference must not rely on the operation's claimed result count. Existing
def-use and linear-consumption checks apply to both words and qubits. A stale
word, duplicated bit, omitted split bit or unreturned zero-width word is invalid.
For width zero, split consumes one owned word and yields no bits, while join
consumes no bits and creates a fresh owned word. There is no physical allocation
or quantum zero claim in either operation.

### Compiler lowering and attestation

Compiler expands register words into ordered tuples of the existing qubit SSA.
Input port ordering is declaration/row-major, and the bit order inside each word
is MSB first, as in the accepted Split/Join wire helpers. Split and join only
rebind tuples. Reordered join operands retain their order, with no emitted SWAP.
An entry register function lowers to one gate function, with explicit flat
qubit arguments and returns. Canonical-payload attestation still runs through
Core's existing gate restoration after lowering. A returned bit permutation
cannot disappear behind an unchanged payload. If callers need a physical output
permutation, that mapping must be explicitly represented and accepted at its
own boundary. Zero-width logical identities disappear from physical tuples only
after their ownership has been verified.

Finite nested calls use invocation-local maps and explicit traversal stacks.
The current `inline_program_calls` gate-only consumer must either lower an
admitted register graph first or refuse it explicitly. It must not accidentally
inline register words as gate operands. Source-free production lowering imports
no upstream graph or numerical simulator and assigns no cost to rebindings.
Existing Add/QROM primitive streams, source dependencies, measurement feedback,
all four complex-amplitude histories, work cleanup and independent cost models
are reused as replacement tests after actual implementation.

### Source witness boundaries

The source builder consumes a scalar zero-width word through Split, records its
instance and produces an empty bit array. Join of that empty array records a new
zero-width word. Shaped empty outputs have no element handles. Duplicate-bit
joins fail after consuming their first input, reserve their instance number and
retain the consumed prefix. A dtype width mismatch can consume the available
prefix before array indexing fails. Construction-time mutation belongs to the
later Compiler builder and is distinct from immutable Core graph validation.
No such failed state may be admitted merely because its source builder exists.

A source classical-value zero-width failure remains separate from a valid
zero-wire graph rebinding. Existing `test_register_split_join.py` already
records that numerical codec boundary, physical permutation amplitudes and zero
bookkeeping gate resources. Reuse those tests without weakening their source
exceptions or claiming numerical zero-width support from structural evidence.


### Prerequisite 1.2 implementation evidence

Core `register_verifier.py` derives types from ordered input ports, resolved
callee output ports and join target descriptors. The existing verifier retains
definition, arity, condition and linearity authority. Immutable function ports,
zero-width ownership, empty arrays, unsupported descriptors, incompatible calls
and returns, physical-profile mixing and payload tampering are checked before
Compiler erases logical words. Default hashes and the original versioned
physical function/operation/block JSON keys remain unchanged.

Compiler `register_programs.lower_register_program` flattens each admitted
function before invoking the existing explicit-stack finite call inliner. It
reuses that inliner's invocation-local maps and DNF specialization. Source-style
mutable graph construction is not implemented by this transformer. Its current
Add/QROM acceptance replaces the existing emitter consumer with real three-level
packed Program functions wrapping the already accepted gate bodies. All original
primitive instructions, dependencies, operands and feedback remain identical.
Both consumers retain all four unnormalized complex measurement histories and
zero workspace. Reordered join outputs also match the original source tensor on
a complex state with a spectator, with one explicit X and no rebinding gates.
Core attestation refuses such a permutation behind an unchanged payload.

The batch adds 59 Core cases, 29 Compiler cases and two contract conformance
cases. The concentrated suite passes 4610 cases without warnings or skips in
26.50 seconds. Eight pinned source blobs are unchanged. Strict Python 3.12
checking covers nine production files, including the existing lowering module.
Black/Ruff, architecture, both team scopes, Program/public API gates and four
existing append/resource examples pass. Persistent evidence is
`ftqc-reference-env/g2-packed-register-{regression,checks,examples}.log` and
`g2-packed-register-source.txt`. Primitive counts and source resource estimates
remain separately recorded in `REGISTER_METADATA.md`.

The 39 original call-builder and 19 rebinding-builder scenarios continue to
execute the source fake. Their mutable construction/failure semantics are still
pending native replacement. Numerical zero-width codec failures, classical and
other unadmitted descriptors, clean/dirty allocation and direct numerical graph
traversal retain their independent acceptance boundaries.

Final focused branch coverage passes all 88 cases. Both new production modules
have 100% statement and branch coverage. The default smoke/unit run stopped
after 5516 passes, 166 skips and 30 warnings at four QFT/rotation fusion
failures. All four reproduce independently from the pre-batch `57081cec`
snapshot. Default-suite acceptance remains open. Logs are
`ftqc-reference-env/g2-packed-register-coverage.log`,
`g2-packed-register-default.log` and `g2-packed-register-baseline-failures.log`.

## Finite register construction prerequisite 1.3

The next Compiler slice constructs concrete word-call functions directly in the
existing Program records. `ComponentProgramBuilder` receives an ordered Signature,
finite immutable function definitions and an entry name. `inputs` groups one
fresh SSA identity per left-port element. `call_dict`, `call_tuple` and `call`
resolve a named typed function and expose its right ports in declaration order.
`finalize` consumes the declared right ports and returns an independently verified
ProgramRecord. No new Core record or machine contract is required.

The existing `create_register_bindings` helper accepts either its original
ValueRef or a zero-argument allocator. An allocator is invoked once per actual
scalar/array element, including a scalar of width zero. Empty arrays invoke it
zero times. Existing fixed-identity callers retain their metadata semantics.
The builder reuses that helper and `process_register_bindings`, including
callback-before-dtype ordering. It snapshots Signature declarations as a tuple.

Each resolved call reserves its operation position before visiting inputs. Its
operand tuple records the consumed prefix even if a later name, index, alias or
dtype check fails. Outputs are allocated only after all input checks succeed.
Returns likewise retain their consumed prefix on failure. Such construction
state is inspectable. Core independently verifies the complete module, including
supplied bodies, at finalization. A failed zero-port call can leave a valid empty
instance, as in the source. An arity-incomplete retained call is rejected by Core.
Unknown names, conflicting definitions and unadmitted executable descriptors are
refused explicitly. Signature-only leaves cannot substitute for numerical bodies.

This slice admits the same concrete quantum bit/opaque/unsigned profile as 1.2.
Automatic allocation/release, classical register handles, builder split/join and
direct numerical traversal keep their own acceptance work. Existing typed split
and join records and gate lowering remain available to real supplied callees.
Source fake cases outside this executable profile retain their source-only
classification. Real Add/QROM emitters remain the replacement consumers, with
all primitive instructions, complete complex branches, workspace cleanup and
independent resource models retained.

Prerequisite 1.3 is implemented by allocator `88abf860` and builder `9da66a0e`.
Core records and machine contract remain at the existing 0.7 boundary. Forty
builder cases, four allocator regressions and 4654 concentrated checks pass.
The two Compiler construction modules have 100% statement and branch coverage.
Existing real Add/QROM consumer instructions, complete measurement histories,
zero work and independent resource models are preserved. Detailed evidence and
remaining admission boundaries are recorded in `REGISTER_METADATA.md`.

## Builder split and join prerequisite 1.4

Extend the same Compiler builder with scalar `split` and rank-one `join`.
A split consumes one owned word and returns a bit-word array. A join consumes
bit-word handles in the requested order and produces one fresh owned word.
Source rank/one-bit guards run before reserving the call. Missing-width array
indices, duplicate bindings and dtype failures retain the same reserved call
position and consumed prefix as the source builder.

`component_rebindings.py` supplies real immutable helper functions in the
existing Program vocabulary. Split helpers use one existing split operation
and one width-one join per output element. Join helpers split each width-one
input word before the existing typed join. Thus the builder exposes uniformly
owned word handles while existing gate SSA remains the lowering authority.
These additional logical rebindings emit zero physical gates and require no
workspace. A width-zero scalar still has one owned handle, while its split
array has no elements. Helper names avoid collisions with supplied definitions.
The helper functions remain Compiler-owned and are verified with the full module.

Reuse the original 19 source witnesses through a native observation adapter.
Original source resource queries stay in their original scenarios. For native
programs derive zero rebinding primitives through actual gate lowering, with
complex output permutations and existing Add/QROM measurement histories checked
independently. This slice does not admit additional descriptor families or
implicit clean/dirty work allocation.

Prerequisite 1.4 is implemented by `488734e1`. Thirty new cases, all nineteen
original rebinding behaviors, two complex spectator/permutation comparisons and
unchanged Add/QROM consumers pass. The fifteen unchanged source scenarios use a
native observation adapter, while four source cost scenarios compare normalized
connections and native lowered zero-primitive cost. The concentrated suite passes
4684 cases. Builder/helper statement and branch coverage are 100%. Source-like
word handle identity uses real helper bodies with extra logical rebindings that
erase to zero physical operations. Numerical graph traversal and automatic work
ownership remain separate. Evidence is recorded in `REGISTER_METADATA.md`.

## Explicit workspace construction prerequisite 1.5

Program contract 0.8 adds `allocate_register` and `free_register` to the existing
finite packed-word profile. Append `Operation.workspace_dirty: bool | None` with
its default excluded from the previous hash tuple. Allocation has zero operands,
one fresh owned register result and an explicit `register_type`. Free has one
owned register operand, zero results and no destination type. Both require an
explicit built-in boolean dirty flag and refuse primitive, condition, physical
or callee metadata. Only allocation/join may carry a register descriptor, and
only allocation/free may carry the workspace flag. Existing def-use and linear
checks enforce consumption even for zero-width words. No physical public schema
or default identity changes.

The pinned source does not match a Free flag back to an Allocate instance.
Input words can be freed, allocated words can become outputs, and a returned
successor can be freed after calls or split/join. Preserve those structural
semantics. Core proves typed ownership only. A clean release imposes a zero-state
obligation and a dirty release imposes restoration of the original joint state.
Neither follows from an accepted graph or from a zero numerical callback.

Compiler `layout/component_workspace.py` supplies real immutable Allocate/Free
function bodies. The existing builder gains `allocate(n=1, dtype=None,
dirty=False)` and `free(binding, dirty=False)` using its ordinary call traversal.
An explicit dtype overrides n as in the source. Scalar zero-width allocations
still reserve a source call and define one owned handle. Reusing a consumed free
input reserves another failed call and retains the source consumption prefix.
The initial executable descriptor and built-in boolean restrictions are explicit
admission checks beyond source construction metadata.

This construction slice must refuse physical gate lowering with an actionable
workspace scheduling diagnostic. It must not erase lifetimes, reset clean bits,
initialize borrowed bits, introduce projection or infer a free work register
from unused gate operands. The next lowering slice must bind lifetimes through
the existing RegisterWorkspace pool, preserve call-local SSA, expose clean and
dirty requirements, and verify full complex states and measurement histories
before claiming automatic workspace execution. Gate-only consumers continue to
use their existing append path. Default physical JSON keys stay fixed.

Source witnesses cover builder identity, source side/width/dirty signatures,
zero-width ownership, reuse/failure timing, classical zero/nonzero behavior,
tensor projections and source resource categories. Native replacement compares
construction with real function bodies and verifies independent Core rejection.
Numerical classical callbacks reuse existing Simulation allocation/free kernels.
Source tensor dirty allocation uses zero while Cirq borrows unknown state. Keep
those source entry points separately identified throughout acceptance.


Prerequisite 1.5 construction is implemented by Core `1b77d7df` and Compiler
`3afb85ef`, following machine contract `daa32e6c`. The concentrated suite passes
4784 cases, and 227 focused cases cover the builder, workspace helper and register
verifier with 100% statement and branch coverage. Ten pinned source blobs remain
unchanged. Physical pool scheduling and clean/dirty numerical restoration remain
separate acceptance work. Evidence is recorded in `REGISTER_METADATA.md`.

## Physical workspace scheduling prerequisite 1.6

The source physical conversion calls `greedy_topological_sort` on each actual
component graph, giving allocation priority +10**16, free priority -10**16,
other concrete components the right-minus-left port width, and stable graph
insertion order as tie breaker. Existing model scheduling groups weak components
for peak estimation. Physical source conversion uses the whole graph. Preserve
that distinction when reusing the scheduling kernel. Do not assume source
builder insertion order equals physical emission order or flatten hierarchy
before preserving each invocation's local schedule.

Eight original-source physical witnesses establish the next execution boundary.
An allocated word may split and free each bit separately. Distinct allocations
may join and release as one word. Source qfree sees the current ordered physical
wires and carries no dirty argument. A clean or dirty double-CNOT lifetime emits
exactly two CNOTs and no allocation/release gates. Complete complex states with
an entangled spectator prove clean work zero and restoration of the original
borrowed joint state. Dirty tensor zero-state callbacks are not the physical
borrowing reference for this evidence.

`layout/workspace.py` remains the physical wire-pool authority. Its existing
whole-register leases are insufficient for source split/merge lifetimes. Extend
that authority with explicit per-wire ownership under partial release, retaining
old whole-lease behavior and foreign/stale/double-release rejection. Release
returns wires to their allocation pool even when the Free metadata differs,
as the source manager has no dirty release flag. Reordering release must remain
observable in later pool allocation. Allocation outputs and freed input words
are valid Core graphs but need explicit caller ownership transfer in a physical
schedule. Until admitted, reject those profiles with a precise diagnostic.

Compiler must consume the verified Program and supplied physical input, clean
and borrowed wire bindings, maintaining SSA values in invocation-local maps.
The output must include the actual primitive order, declared wires, returned
port bindings and explicit allocation/release events. Reuse existing Core
records and finite call traversal. State simulation remains Simulation-owned,
and models remain independent queries. The existing gate-only lowering remains
valid for programs without lifetimes.

Acceptance must independently compare original source graph traversal and
manager events, physical operands, feedback, output order and all small
measurement branches. Test interleaved allocations, partial releases, cross-
allocation joins, zero-width lifetime identities, failures, dirty restoration
with external entanglement and clean-work zero. Resource evidence must list
primitive emissions, declared versus live workspace and source model categories
separately. A schedule is ownership evidence, not a numerical cleanup proof.

The eight source-only witnesses pass in 1.95 seconds in
`tests/unit/test_register_workspace_schedule_contract.py`. They are prerequisites,
not completed native scheduling or additional accepted capability counts.

### Physical binding entry and local connection ordering

The first physical entry is Compiler-internal `bind_register_program(record,
inputs, *, clean=(), borrowed=(), entry="main")`. Inputs are ordered physical
wire tuples, one per declared left element. Return a frozen Compiler result with
the existing `CircuitIR`, right-element wire tuples and immutable lifetime events.
Each event records its flattened operation position, kind, ordered wires and
source dirty metadata. These records serve the concrete binding and cleanup
consumer. They do not attest numerical restoration. Existing gate-only lowering
keeps its current result and semantics.

For each actual function, reconstruct connections by consuming operand ports in
operation order and then return ports in declaration order. Insert source and
target nodes when an edge is first seen, as the source `_create_binst_graph` does.
Parameter and feedback SSA introduce zero-width dependencies. Helper functions
whose sole operation is Allocate/Free inherit that source scheduling priority.
Keep every call's local ordering before finite expansion, reusing the existing
explicit-stack call traversal. Unconnected source nodes originate in a frozenset
and have no portable insertion tie. Native unconnected operations use declaration
order. That is a deterministic admitted policy and needs a separately disclosed
source tie limitation rather than an exact-order claim for that case.

The first physical profile requires quantum-only entry arguments, supplied
nonnegative built-in integer wires, disjoint input/clean/borrowed bindings and
balanced physical workspace. Freeing caller inputs and returning nonempty live
workspace require explicit ownership transfer and are refused. Register views
and call-local aliases propagate the actual physical tuple. Partial free returns
each wire to its original pool, independently of Free metadata. Zero-width
logical identity is validated by Core before physical tuples are erased.

Emit constants, classical parity, measurements, reset, primitive parameters and
boolean feedback through existing Instruction and CircuitIR contracts. Preserve
fresh measurement slots per actual invocation. The caller supplies initial clean
and dirty states and receives unchanged wire numbering, including unused declared
wires. Graph binding does not insert numerical cleanup. Exercise the entry in an
unchanged circuit.append consumer by replacing its emitter and test native
complex states plus all small measurement branches. Input/output transfer,
classical entry ports and unconnected source node ties stay explicit limitations.

### Allocated output ownership transfer prerequisite 1.7

Extend the internal physical binding entry with keyword-only
`allow_workspace_outputs: bool = False`. Require a built-in boolean. The default
retains balanced physical lifetimes. An explicit true value transfers any
returned nonempty allocated wires to the caller through the existing ordered
`outputs` result. Existing allocation events retain each wire's clean/borrowed
origin and source flag. Returned work stays live, receives no artificial Free,
and remains unavailable to later allocations within the same binding operation.

Core's existing linear ownership verifies that live wires reach declared right
ports. Transfer includes current split/join permutations, mixed input/allocated
output words and nested call results. Entry port order and MSB order remain the
same. A zero-width right word retains its logical Core identity and returns an
empty physical tuple. The existing CircuitIR still requires at least one
physical wire. No new Core record, serialized field or public export is needed.

The caller now owns the output lifetime, including subsequent cleanup and final
return of borrowed work. Binding preserves the exact state produced by the
component. A borrowed output may remain entangled or transformed until its
caller completes the enclosing algorithm. Numerical acceptance compares the
whole source physical state, returned wire order and measurement branches.
Source tensor allocation still starts from zero and is a different entry.

Eight unchanged-source witnesses in
`tests/unit/test_register_output_transfer_contract.py` pass in 2.67 seconds.
Clean/borrowed right-only words of widths 0, 1 and 3 preserve reversed wire order,
emit no gates, and cause allocation without a free event. Source QEC gate cost
is empty and modeled peak includes data plus the right-only word. Two source
default-manager witnesses reject freeing unmanaged external LineQubit inputs.
Caller-input release remains a separately declared pool-ownership boundary.

Compiler acceptance must retain the existing balanced-profile rejection, reject
non-boolean policy values, and prove physical output transfer using real function
bodies through an unchanged append consumer. Compare source primitive operands,
all small unnormalized complex branches and joint spectator states. List output
lifetime ownership, emitted resources and modeled peaks separately. The source
contract witnesses alone do not certify native transfer execution.

### Pre-owned input pool prerequisite 1.8

Eight source witnesses in `tests/unit/test_register_input_pool_contract.py`
pass in 5.24 seconds. A real GreedyQubitManager first owns the input through
qalloc/qborrow. The component reverses that input, frees it and returns a new
right-only allocation. Original local greedy traversal frees the input before
allocating the output. With no other available wires, the output reuses exactly
the released physical tuple for widths 1 and 3 and both allocation/Free flags.
The manager receives no dirty argument on free. Primitive emission is empty,
source QEC cost is empty and modeled peak equals width.

These witnesses complement prerequisite 1.7's default-manager refusal of
unmanaged external inputs. Neither result permits implicit input adoption.
The next internal binding extension is keyword-only
`input_ownership: tuple[tuple[int, bool], ...] = ()`. Each pair declares an input
physical wire already owned by the caller's clean or borrowed pool, with the
built-in boolean representing dirty origin. All pairs must be immutable,
distinct, valid input wires with built-in boolean flags. Empty declarations
retain unmanaged input behavior. A partially declared input word cannot release
an undeclared wire. Validate the complete release before mutation.

Extend the existing RegisterWorkspace with atomic `adopt(wires, *, dirty=False)`
returning its existing WorkspaceLease. Adoption registers an existing lifetime.
It emits no Program Allocate event, consumes no available workspace, and refuses
collisions with either available or active wires. Reuse existing pool ownership
and partial release. Input leases must be distinguished from allocations created
inside the Program, so unchanged returned inputs remain ordinary caller outputs.
Released input wires return to their declared origin pool, independently of the
source Free flag, and may subsequently be allocated through the ordinary path.

Core retains its current logical single-consumption authority, including
zero-width values. Compiler declares physical pool ownership only. The existing
balanced-profile rule concerns newly allocated nonempty work. Existing
`allow_workspace_outputs` governs any newly allocated work that escapes, even
if it reuses a previously released input wire. Adoption alone does not authorize
arbitrary workspace outputs. Free still requires the emitter/caller's clean-zero
or borrowed-joint-state restoration obligations at its actual lifetime boundary.

Acceptance must compare original source manager events and output tuples, prove
exact complex spectator states, all small measurement branches, per-wire pool
origin and subsequent reuse. Include both Free flags, mixed input/allocated
words, partial ownership failure without release prefixes, live returned inputs,
policy validation and retry safety. Preserve the existing pool's established
reuse policy. The source GreedyQubitManager witness above has no spare available
wires and does not certify parity with every possible manager reuse policy.

This is a contract and source prerequisite. Native pool adoption and the new
binding keyword have not been implemented at this commit. No migrated capability
checkbox closes on this evidence alone.

### Classical owned-port prerequisite 1.9

The next boundary starts with actual locked-source classical register paths,
not a synthetic quantum-port fake. Four contract witnesses in
`tests/unit/test_register_classical_port_contract.py` cover CBit's zero
quantum and one classical bit, a two-element Parity input and right output,
MeasureZ's quantum-left/classical-right transition, and the current native
executable-word refusal. Parity's original classical call retains the input
array and returns its XOR for every two-bit value. MeasureZ's original model
reports one measurement and peak one qubit. With a correctly shaped CBit input,
the locked Parity Cirq path raises a zero-qubit arity error while attempting to
wrap classical slots as qubits. The source explicitly refuses classical right
wires later in the same adapter. This is a source Cirq-export limitation, not
evidence of an executable classical-port circuit.

The internal Core representation already has the exact nominal descriptor
`RegisterType("bit", 0, 1)` and a linear `register` value kind. The
existing `bool` kind remains a shareable measurement-feedback SSA value.
An owned CBit port must not alias that `bool` value or inherit its fanout.
The first implementation step must admit concrete CBit ports in the register
verifier without changing quantum word width rules or the protected public IR.
The source MeasureZ transition needs an explicit internal operation consuming
one quantum resource and producing one owned CBit handle. Its physical binding
emits one existing measurement instruction and associates the output handle
with that measurement's classical slot. It does not manufacture a successor
qubit, reset the measured wire, or treat a classical bit as a physical qubit.

Physical result metadata must distinguish ordered quantum-wire outputs from
ordered classical-slot outputs while retaining the existing CircuitIR and
unchanged user `circuit.append(op)` path. A classical handle passed to a
later component stays single-consumption; any parity or feedback uses must
explicitly preserve the source's output ownership and consume the allocated
classical slot. The builder must reject stale handles and width/type mismatches
through the existing source-ordered validation. A source-only classical input
requires a separately verified executor initialization contract before physical
binding. The first executable slice may therefore accept measurement-produced
CBit values and fail closed on external classical entry ports.

Acceptance must compare source graph wiring, CBit arrays and XOR value order,
measurement-slot dependencies, all two-outcome unnormalized complex branches,
quantum work status and separate emitted/model resources. Retain the locked
source Cirq-export failure as a boundary case. Do not claim a flat source gate
stream for that unsupported export path. This is a contract/source prerequisite;
native classical-port execution and its capability checkbox remain open.

#### Owned CBit port admission

Core now accepts the concrete existing `RegisterType("bit", 0, 1)` descriptor
at finite function ports. Each scalar or row-major array element has its own
linear `register` identity. Existing nested identity calls and the existing
Compiler builder preserve single consumption. A CBit cannot substitute for a
shareable `bool` feedback value. Quantum width inference, split/join and
workspace allocation/free continue to reject classical descriptors.

Execution compatibility additionally requires equal classical widths before
applying the source's loose metadata predicate. That predicate intentionally
accepts CBit versus opaque zero-qubit metadata. It cannot authorize erasing a
classical resource at a call or return boundary. The metadata predicate retains
its source behavior, and both directions of this execution refusal are tested.

The acceptance suite includes two original CBit-to-CBit Cast calls, numerical
values zero and one, stale-handle refusal, mixed classical/quantum arrays,
zero-width quantum identities and malformed classical descriptors. The
27 new cases and four source prerequisites are included in 4,977 concentrated
passing tests. A 203-test focused run measures 98.19% statement/branch coverage
of `register_verifier.py`. Nineteen locked source blobs are unchanged. Strict
typing covers the same 17 production modules as GRAPH25, alongside format,
architecture, Program/API, scope and four existing append/resource examples.
This slice adds structural verification. Measurement-produced handles, native
classical evaluation, physical-slot outputs, branch amplitudes and emitted
classical resources still require the next implementation and acceptance.

The final annotation-only follow-up is `073f4f44`. All 4,977 concentrated tests
pass again at that head in 76.14 seconds. Strict typing also passes for the new
conformance file. The default smoke/unit tier was restarted with four workers
after 4,991 serial passes. Its bounded fail-fast run stops with 22,286 passes,
300 skips, 1,065 warnings and seven failures in 391.44 seconds. All seven
failures reproduce in an independent complete `45fe5f67` snapshot with tracked
tree `6198e732324634286d01ef3a3da4665a2f0b9663` in 52.20 seconds. They cover
terminal rotation fusion, the local compact-CX threshold, QFT composition and
four qubit-vocabulary ledger assertions. The full default tier remains open.
Logs are `ftqc-reference-env/g2-classical-port-default-parallel.log` and
`g2-classical-port-baseline-failures.log` in the external evidence directory.
The scope audit refresh changes only this implementation's fingerprint.

### Destructive owned measurement prerequisite 1.10

The internal `measure_register` operation consumes one owned register with the
exact QBit descriptor `RegisterType("bit", 1, 0)` and produces one owned register
with `classical_bit_type()`. It carries no parameters, condition, literal,
physical payload, callee, register descriptor or workspace flag. Core derives
its result type and applies existing SSA single-consumption verification.
This operation is confined to declared register functions.

Compiler binds it to one existing dynamic `measure` instruction with one fresh
classical slot. The input quantum handle has no successor. Finite nested calls
carry the output slot by the existing SSA remapping. Scheduling uses zero quantum
width for CBit edges while retaining their dependency edges. The separate gate
SSA lowering continues to reject classical ports.

`BoundRegisterProgram.outputs` retains one quantum-wire tuple per returned
logical element. A classical element occupies an empty quantum tuple. A trailing
internal field `classical_outputs`, defaulting to `()`, contains parallel
per-element slot tuples whenever any classical element is returned. Quantum
positions have empty slot tuples. Pure quantum callers retain their current
three-argument construction and empty classical metadata. External CBit entry
initialization remains outside this profile and is rejected before scheduling.

This first binding admits destructive measurement only on caller quantum inputs
without pool adoption. A wire in a live or adopted workspace lease is rejected
before emitting its measurement. Neither output-transfer policy nor a dirty flag
authorizes destructive consumption of borrowed state. The pool receives no free,
reset or fabricated restored successor. A separate ownership contract is needed
for destructive workspace retirement.

Source conformance uses the unchanged MeasureZ copy tensor and its DiscardInd
leg. For each recorded outcome the tensor gives the unnormalized complex
instrument on the retained spectator. Native branch checks project the measured
physical wire, then compare its fixed-outcome sector with the source instrument.
The measured wire remains physically projected and unavailable through the
consumed handle. It is not asserted to be zero. Unmeasured clean workspace must
still be returned zero. Emitted measurement counts and source QEC model costs
are recorded separately. The source classical-port Cirq export limitation stays
open. This prerequisite does not accept native execution by itself.

#### Destructive measurement acceptance

Contract `d4e6786a` precedes Core `2ca953bc` and Compiler `35cd0b68`.
Twenty-four new source, Core and binding cases join 5,001 concentrated passing
tests in 22.38 seconds without skips or warnings. A 212-test focused run covers
all statements and branches of physical binding and local scheduling, with 99%
aggregate coverage including register verification. Twenty source blobs match
the locked inventory. Strict typing covers 17 production files and the three
native conformance test files, alongside Black/Ruff, scope, architecture,
Program/API gates and four existing append/resource examples. Local integration
passes 29 tests. Both actual dynamic runtime strategies additionally match 64
normalized complex conditional states each, observing all four histories and
zero unmeasured work through `circuit.append(op)`.

The default smoke/unit fail-fast run reports 22,286 passes, 300 skips,
1,065 warnings and seven failures in 117.28 seconds. Its failure identifiers
exactly match the prior independently reproduced `45fe5f67` baseline failures.
The default tier remains open. Original classical-port Cirq export, external
CBit initialization, owned classical parity/feedback and destructive workspace
retirement still require their own acceptance. No complete source file closes
on this slice. Evidence is retained under `ftqc-reference-env/` with prefix
`g2-owned-measurement-`, including regression, coverage, checks, examples,
integration, runtime, default-parallel and source-integrity records.

### Owned classical parity and feedback prerequisite 1.11

The internal `parity_register` operation consumes N owned CBit elements in
operand order and returns N owned successors followed by one new owned parity
CBit. N is the immutable operand count, including zero. All operands require
`classical_bit_type()`. Core supplies the variable-arity signature and derives
all result types. Existing SSA rules reject duplicate inputs and stale owners.
No operation metadata, parameters, conditions or workspace flags are admitted.

Compiler constructs a finite `parity_word_function(name, count)` with CBit[count]
THRU `x` and scalar CBit RIGHT `parity`, following locked Parity's signature.
This executable profile requires a built-in nonnegative count. It binds each
successor to its predecessor's immutable classical slot and emits exactly one
existing `ClassicalXor` assignment to a fresh slot, including the zero-input
constant-zero assignment. Source Parity has no executable decomposition or QEC
cost implementation. Its original classical callback and nominal signature are
the comparison authority. A native classical XOR is recorded separately from
source quantum model availability.

The internal `read_classical_register` bridge consumes one owned CBit and
produces an owned CBit successor plus a shareable `bool` snapshot. The successor
and snapshot refer to the same immutable slot. Binding allocates no slot and
emits no instruction for the bridge. Fresh slots are never overwritten within
this binder. The bool can feed existing conditional quantum gates and bool XOR,
while the owned successor remains singly consumed. This bridge is a native
execution adaptation with no claim of an additional source gate or model cost.
It carries the same metadata restrictions as parity. It is confined to register
functions, and its result cannot turn into an external bool boundary port.

Both operations require physical register binding and fail closed in quantum-only
SSA lowering. External classical initialization and destructive workspace
retirement retain their separate prerequisites. Numerical execution continues
through existing CircuitIR and `circuit.append(op)` on the local single-device
path. Acceptance covers zero/one/many inputs, nested parity calls, fresh slots,
classical ownership failures, snapshot fanout and both condition polarities.
For measurement-produced inputs, compare every small unnormalized complex branch
against the unchanged MeasureZ tensor and original Parity callback, including
spectator phase and zero unmeasured work. Record emitted measurements, classical
XORs, conditional quantum primitives and available source models separately.

#### Owned parity and feedback acceptance

Contract `b8f3cb8d` precedes Core `2061cf1c` and Compiler `380ea648`.
Fifty added cases cover the unchanged Parity callback on every zero-through-four
bit input, Core ownership and shape failures, nested parity slot dependencies,
both control polarities, bool fanout and quantum-only lowering refusal. The
measurement-to-feedback scenario compares all 30 unnormalized branches for
zero through three measured inputs against original MeasureZ tensors, original
Parity callbacks and the original ClassicallyControlled Z/X tensors. Joint
spectator phases and zero unmeasured work are retained. The existing append
path also executes 32 trajectories per polarity and runtime strategy, with
128 normalized conditional complex states checked in total.

The final concentrated regression passes 5,051 tests in 41.64 seconds without
skips or warnings. The focused 254-test run covers all statements and branches
of the classical component constructor, physical binder and local scheduler,
with 99% combined coverage including register verification. Twenty-six source
blobs match the locked inventory. Strict typing passes for 18 production files
and five native test files. Black/Ruff, scope, architecture, Program/API gates,
four existing append/resource examples and 29 local integration tests pass.

The default smoke/unit fail-fast run reports 22,286 passes, 300 skips,
1,065 warnings and seven failures in 217.46 seconds. Failure identifiers exactly
match the prior independently reproduced baseline. The full default tier remains
open. Source Parity's QEC query and decomposition still raise the original
DecomposeNotImplementedError. Its zero QubitCount is separate from the native
one-XOR-per-call classical resource count. The combined source graph has no
complete QEC model on this evidence.

Owned classical discard, the actual GF2MulMBUC owned-component replacement,
external CBit initialization and destructive workspace retirement remain open.
Evidence uses prefix `ftqc-reference-env/g2-owned-parity-` with regression,
coverage, checks, examples, integration, default-parallel and source records.

### Owned classical discard prerequisite 1.12

The internal `discard_classical_register` operation consumes exactly one owned
CBit and produces no results. Only `classical_bit_type()` is admitted. Existing
SSA linearity rejects stale owners, duplicate consumption and returning a
consumed owner. The operation carries no parameters, conditions, literal,
physical metadata, callee, workspace flag or register descriptor.

Compiler supplies `discard_word_function(name)` with one scalar CBit LEFT `c`,
following the unchanged Qualtran v0.7.0 `basic_gates/discard.py` Discard signature.
There is no bit order conversion. Binding removes only the consumed owned
handle. Previously exposed shareable bool snapshots retain their immutable slot,
and future assignments use fresh slots. The operation emits zero instructions,
allocates no slot and releases no physical quantum wire. Quantum-only lowering
requires physical register binding. External CBit initialization and physical
workspace retirement retain their separate contracts.

The source classical callback returns an empty mapping for either bit value.
The source tensor contributes `DiscardInd` on the incoming classical index.
Acceptance checks this source behavior, complete unnormalized measurement
branches before tracing out the classical history, spectator phases, clean
unmeasured work, both feedback polarities and feedback after owned discard.
Actual primitive counts and available source resource queries are recorded
separately. The numerical path uses existing `circuit.append(op)` execution.

#### Owned classical discard acceptance

Contract `4d2881d8` precedes Core `6b697df2` and Compiler `409f724d`.
Thirty-one added tests cover the unchanged Discard callback, signature,
DiscardInd tensor marker and resource behavior, malformed operations, stale and
double consumption, and feedback after an owned handle is consumed. Zero through
three measurements and both feedback polarities compare all 30 unnormalized
complex branches against unchanged MeasureZ and controlled Z/X tensors and
Parity callbacks. The reduced target/spectator density matrices also agree after
summing discarded histories. Unmeasured work remains zero. Both actual dynamic
runtime strategies check 32 conditional states per polarity, totaling 128.

The concentrated regression passes 5,082 tests in 27.64 seconds without skips or
warnings. The focused run passes 285 tests in 7.18 seconds, covering every
statement and branch of component construction, physical binding and scheduling.
Combined coverage with register verification is 99%. Twenty-seven source blobs
match the locked inventory. Strict typing passes 18 production and eight native
test files. Formatting, scope, architecture, Program/API gates, four existing
append/resource examples and 29 local integration tests pass. The generated
scope audit changes only the register verifier source digest.

Classical discard emits zero instructions, allocates no classical slot and has
zero quantum width. Source QECGatesCost also returns zero, while source
Discard.decompose_bloq() raises DecomposeNotImplementedError. The complete
measurement/parity/feedback path still has no original Parity QEC model. These
observations keep emitted resources separate from model availability.

The actual GF2MulMBUC owned-component graph, external classical initialization,
destructive workspace retirement and prerequisite Q-0157.03 remain open.
Evidence is retained under `ftqc-reference-env/g2-owned-discard-` with regression,
coverage, checks, examples, integration and source-integrity records.

The default smoke/unit parallel fail-fast run reports 22,264 passes, 300 skips,
1,065 warnings and six failures in 117.80 seconds. All six failure identifiers
are a subset of the seven previously independently reproduced baseline failures.
The remaining baseline ledger failure was not reached before interruption.
The full default tier remains open. Exact failure comparison and the default
log use the same evidence prefix. This run adds no default-tier acceptance claim.

### Binary-field execution prerequisite 1.13

Existing finite_field descriptors may enter the finite register executor when
quantum width is a positive built-in integer, classical width is built-in zero,
and ordered neutral parameters retain characteristic 2, degree equal to width
and order 2**width. These three fields are followed either by no polynomial
identity or by the existing complete five-field polynomial identity. Every
parameter value is a built-in integer. Core checks this structural profile.
Numerical polynomial construction and irreducibility remain at their existing
owners. No external library object enters Core.

Split and Join preserve the exact descriptor, port identity and MSB-first bit
order. Equal-width fields with distinct nominal polynomial metadata remain
subject to the existing source-compatible dtype checker. Other field profiles
remain outside finite execution. The existing physical binder and quantum-only
word lowering reuse executable_register_width without a second field codec.
Acceptance compares unchanged source Split/Join and the existing native codec,
then verifies physical permutation, complex spectator state, workspace cleanup
and zero emitted-gate/model counts for pure rebindings.

### Owned X measurement prerequisite 1.14

A finite measure_x_word_function composes existing split, H, join and destructive
measure_register operations, with QBit LEFT q and CBit RIGHT c. It adds no Core
opcode. The logical input is consumed once. Physical binding emits H followed by
one Z measurement into a fresh immutable slot. The physical measured bit retains
its projected outcome. Existing clearing emitters retain their distinct
conditional-X behavior. Pool-owned destructive measurement keeps its existing
retirement refusal. Source MeasureX's tensor is a density instrument with a
DiscardInd trace marker, so compare its branch density matrix and the native
unnormalized branch vector using the same H/Z factorization, including joint
spectators and zero unmeasured work. Model queries and emitted primitives are
recorded separately.

### Owned binary-field product erasure prerequisite 1.15

product_erasure_program(dtype) composes existing register Programs, accepting a
resolved binary field with an explicit irreducible polynomial over GF(2).
The shared Compiler polynomial validator owns concrete irreducibility checks.
The function preserves original x/y THRU ports and consumes result LEFT.
It follows GF2MulMBUC's reversed split order, MeasureX calls, low product CZ
corrections, reduction-matrix column parity calls, high product CZ corrections,
per-parity discard, final measured-bit discard and reversed joins.

The reusable classical_cz_word_function forwards one owned CBit, exposes a
shareable snapshot and applies existing conditional CZ to two owned QBits.
There is no new primitive, public API or resource identity. Each Parity gets a
fresh slot instead of the flat clearing profile's explicitly reusable slot.
The graph emits width H, width measurements, width classical XOR assignments
and width**2 conditional CZ gates. The unchanged original QEC call graph reports
only width**2 CZ. Its omitted measurement and classical costs remain explicit.
The source reports no complete executable Cirq decomposition for classical
ports. Source build_composite_bloq runs unchanged against native owned bindings
as a replacement test and must yield the same connections and operands as the
native builder. Actual scheduler order is checked separately from source
construction order, retaining all data dependencies.

Destructively measured product wires retain projected computational outcomes.
Their reduced logical input state matches every original erasure branch after
tracing these consumed wires. Unmeasured private work stays zero. The existing
flat emitter retains physical clearing behavior. Workspace retirement and
caller-owned clearing of destroyed graph wires require separate acceptance.

#### Owned field erasure acceptance

Contract `0da7d70f`, binary-field Core `6dac1163`, composition contract
`5877e729`, shared measurement/feedback `d3f309b2` and field graph `fdcd859c`
are separate commits. Thirty-four added tests verify concrete binary-field
admission and refusal, unchanged source Split/Join codecs, MeasureX joint
spectator branch tensors, source graph replacement, exact branch phase and
resources. The unchanged GF2MulMBUC.build_composite_bloq method directly builds
with native owned bindings and yields an identical Program for (degree,
polynomial) pairs (1,3), (2,7), (3,11), (3,13), (5,37), (8,283), (8,285).
This compares every declared component connection, operand and successor.

All 14 histories for degrees one through three preserve complete complex input
amplitudes and spectator entanglement, with the exact unnormalized factor
2**(-degree/2). Destroyed result physical wires retain the recorded Z outcomes.
Unmeasured work stays zero. Both actual dynamic runtime strategies additionally
check 32 normalized conditional states each through circuit.append(op).
Owned workspace retirement continues to fail closed on the original boundary.

The eight-bit graph emits 8 H, 8 measurements, 8 classical XOR assignments and
64 conditional CZ gates. It emits zero conditional-X clearing gates. Its eight
parity outputs get distinct fresh immutable slots. Original GF2MulMBUC QEC cost
reports 64 CZ and omits measurement and classical costs. Original MeasureX
individually reports one measurement and zero T. Source construction order and
Compiler's connection-dependent physical schedule are distinct observations.
The existing flat erasure implementation retains physical clearing and its
caller-owned reusable parity slot.

The concentrated regression passes 5,116 tests in 114.19 seconds without skips
or warnings. Focused coverage passes 319 tests in 28.78 seconds, with 100%
statement/branch coverage for all six Compiler modules measured, including the
three new constructors and erasure graph. Combined Core/Compiler coverage is
99%. Strict typing passes 21 production and 11 native test files. Black/Ruff,
scope, architecture, Program/API gates, four existing append/resource examples
and 29 integration tests pass. Twenty-seven pinned source blobs are unchanged.
The generated scope audit updates only the register verifier source digest.

This acceptance covers the concrete binary-field owned graph. Source classical
port Cirq export, external CBit initialization, physical retirement and the
broader Q-0157.03 replacement inventory retain independent acceptance. G2 remains
open. Evidence uses prefix ftqc-reference-env/g2-owned-field-erasure-.

The default parallel smoke/unit fail-fast run reports 22,455 passes, 300 skips,
1,065 warnings and nine failures in 325.46 seconds. Seven identifiers exactly
match the previous independently reproduced baseline. Two additional failures
come from sandbox denial of local socket binding in multinode preflight tests.
Both pass in 0.18 seconds when rerun individually with local socket permission.
The original run, permission-specific rerun and exact failure comparison are
retained under the same evidence prefix. The full default tier remains open.

### Measured workspace retirement prerequisite 1.16

The internal physical binder accepts measured_workspace with values reject
(default), retire and reset. This is a physical lifetime policy applied only
when destructive measure_register consumes pool-owned wires. Ordinary caller
inputs retain their existing measurement behavior. Validate the policy before
constructing a binding. Existing logical Program operations remain authoritative.

Retire removes a clean measured wire from its lease without returning it to the
available pool. Its physical outcome remains projected and its event is retire.
Reset emits one existing reset primitive after the measurement, returns the
clean wire to its original pool and records a free event. The fresh measurement
slot survives either operation, including when an owned CBit is discarded after
creating a bool snapshot. A reset does not allocate a new logical feedback slot.
Partial leases retain all other live wires and their original release order.
Allocated and explicitly adopted clean inputs use the same lifetime rules.

Both explicit policies reject borrowed wires before changing pool state or
emitting a measurement. Resetting a borrowed wire cannot restore its unknown
initial joint state. All requested wires must have current owners and the
complete request is validated before any partial release. Retired wires cannot
be freed again, returned as quantum outputs or silently reused by an allocation.
Zero-width bookkeeping continues to use the existing whole-lease contract.

Pinned Allocate and Free emit no quantum instructions and delegate bookkeeping
to the source qubit manager. Free's clean tensor assumes a zero input and its
dirty contract requires the original state. Neither supplies destructive
retirement. The explicit policies above are FlagQuantum physical adaptations,
with reset overhead recorded separately from source measurement and QEC models.
Acceptance must cover every small complex branch, joint spectator amplitudes,
physical zero after reset, projected outcomes without reset, stale and partial
owners, rejected mixed clean/borrowed requests, bool snapshots and exhaustion.

#### Measured workspace retirement acceptance

Contract 69a1383f precedes pool 138d5ad8 and binding 61ec15e6. Forty-seven added
tests pass. The concentrated suite passes 5,163 tests in 43.90 seconds without
skips or warnings. Focused coverage passes 408 tests in 17.28 seconds with 100%
statement and branch coverage for all seven measured Compiler modules, including
the pool and physical binder. Combined measured Core/Compiler coverage is 99%.
Twenty-nine integration tests pass in 1.72 seconds. Strict typing passes 34
production/test files, with Black/Ruff, scope, architecture, Program/API checks
and four existing append/resource examples passing. All 27 pinned source blobs
remain unchanged. Evidence uses ftqc-reference-env/g2-workspace-retirement-.

Every small branch of the original MeasureZ instrument with joint complex
spectators, discarded-owner snapshots and both feedback polarities matches the
explicit physical policy. Both runtime strategies verify 128 conditional states.
GF2MulMBUC additionally passes all 14 histories for widths one through three
under each policy, retaining exact phase and unnormalized amplitude. Reset
clears every destroyed result wire. Retire retains each projected outcome.
The eight-bit stream retains 64 conditional CZ, 8 H, 8 measurements and 8
classical XOR assignments, with exactly 8 additional resets in the reset profile.
Original GF2MulMBUC QEC reports only 64 CZ. Reset overhead is a native physical
policy cost and is not attributed to the source decomposition.

The default smoke/unit tier was not rerun in this batch. Its previously recorded
seven independent baseline failures and permission-specific socket recheck remain
recorded in GRAPH30. External classical initialization, Q-0157.03 full source
replacement inventory and general borrowed-state restoration remain open.
