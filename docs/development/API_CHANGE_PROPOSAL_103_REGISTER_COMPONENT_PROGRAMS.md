# Proposal 103: Register component programs

## Status and scope

Draft integration proposal for G2 of PR #1084. The metadata prerequisite 0.1
below is implemented independently of the proposed Program IR extension. This
document does **not** declare owner approval, change the existing Program
contract, or claim register graph execution is implemented.

The repository's Core instructions require an integration contract/ADR change
before changing IR. The existing Program contract is version 0.5 and admits
gate-level SSA, canonical dynamic conditions and a finite physical profile.
Its approval of dynamic lowering does not establish register call semantics.

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
