# Binary field inverse whole-file audit

This batch covers Q-0515, Q-0516 and Q-0517 at frozen Qualtran v0.7.0 commit
`bcb17bb25360ed68ce3bc084645db3e929799389`, with directly required field
addition, Frobenius, multiplication and register/workspace dependencies.

## Ownership and interfaces

Compiler owns the existing source-ordered raw inverse decomposition, exact
compute-copy-uncompute XOR wrapper, caller-owned workspace and independent
source declarations. Simulation owns field values and the original result/junk
callback. Examples compose these owners with `circuit.append(op)`. These are
single-device correctness paths using existing APIs and IR contracts.

Field words are big-endian. Raw input is THRU, result and junk words are RIGHT.
The caller provides clean result/junk/scratch storage. Only recursive scratch
returns to zero in the raw computation. Junk remains a logical output. The
clean XOR wrapper accepts an arbitrary target, retains the source and restores
all workspace to zero. Physical output maps are distinct from free logical
labels. Measurements belong to the separately accepted fixed-wire inverse
chain and its erasure owner.

The complete source inventory includes one class, its constructor and declared
field, six cached properties, four further operational methods, two factories
and one documentation specification. All six original test functions and the
complete eleven-cell notebook are audited with their direct dependencies.

## Complete definition disposition

| Original definition | Native ownership and disposition |
|---|---|
| `qgf`, `__init__` and its converter/deprecated `bitsize` path | Existing Simulation `binary_field_encoding` and neutral register metadata supply degree/polynomial construction. A foreign Bloq constructor and its deprecated keyword are outside the native interface. Generic constructor/serialization admission remains open. |
| `signature`, `bitsize`, `n_junk_regs`, `bitsize_hamming_weight` | `field_inverse_catalog` and Compiler `inverse_model` retain all port names, directions, shapes and degree-one omission. Degrees 1–16 and both original symbolic assumptions match the unchanged source. |
| `_bits`, `gf2_multiplier`, `build_composite_bloq` | Existing `field_inverse` owns the complete Itoh–Tsujii chain, recursive Karatsuba calls, logical output maps, scratch ownership and exact linear adjoints. Accepted `4e6aa871` gate/operand/output evidence is reused. |
| `build_call_graph` | Independent `inverse_source_calls` retains every concrete square power and adjoint identity, addition/product multiplicity and symbolic shape-only linear child. Complete one-level edges and leaf counts match degrees 1–16 and both symbolic instances. |
| `my_static_costs` | Neutral signature right-side bit totals match `QubitCount`, including THRU x. Static output size and actual allocated workspace are separate accounts. |
| `on_classical_vals` | Simulation `finite_field_inverse` retains scalar assertions, zero mapping, original intermediate junk values and exact NumPy empty-array dtype. Every element at degrees 1–8 and selected elements at 9/12/16 match. |
| `_gf16_inverse`, `_gf2_inverse_symbolic`, documentation specification | Concrete degree four and positive-integer symbolic m have full native signatures/declarations. The notebook's unrestricted m is also preserved, with its distinct original bit-width expression. |

The shared multiplication model previously expressed the symbolic Karatsuba
exponent as exact `log(3)/log(2)`. The frozen source evaluates numeric `log2(3)`
to a floating value before exponentiation. The shared owner now preserves that
original expression. Direct inverse QEC comparisons cover both symbolic forms.
No gates are emitted to derive declarations or symbolic costs. Concrete square
costs come from the independently owned numerical matrix and PLU cost formula.

## Whole original test disposition

| Original function | Native checks and remaining boundary |
|---|---|
| `test_gf16_inverse` | Full degree-four signature/graph, scalar/raw complex outputs, exact reversal and existing clean XOR/gate-stream checks. Generic foreign serialization remains open. |
| `test_gf2_inverse_symbolic` | Positive-integer source factory and unrestricted notebook symbol both retain full signature, junk shape, static width, graph and QEC expression. Symbolic gate emission remains outside concrete admission. |
| `test_gf2_inverse_symbolic_toffoli_complexity` | Exact original symbolic Toffoli expression and right-side static width assertions pass. Independent native models match source QEC at degrees 1–16 and both symbolic forms. |
| `test_gf2_inverse_classical_sim_quick` | Degree-one source behavior is checked unchanged. Native raw output has no junk port. The numerical callback's empty junk array is retained. |
| `test_gf2_inverse_classical_sim` | All original slow degrees 2/3/4/5 complete. Stronger scalar coverage at degrees 1–8 and existing clean execution checks are reused. |
| `test_gf2_equivalent_bloq_counts` | Original degree 1–16 declaration/decomposition count assertions complete. Native complete graphs and independently composed models match. Source linear-adjoint defects remain separately registered. |

The untouched original full test file, including slow cases, reports **27 passed,
3 skipped and 2 xfailed** in 1.28 seconds. Its three skips concern symbolic-shaped
decomposition, symbolic counts available only through the declared call graph,
and dtype checks requiring decomposition. Both xfails concern unsupported
example serialization. One warning concerns the reference's unregistered slow
marker. None of these outcomes is reported as native acceptance.

## Complete teaching asset

`examples/notebooks/binary_field_inverse.ipynb` preserves every topic of the
original eleven cells, including recurrence, parameters, registers, references,
both original examples, concrete graphical signature and full one-level graph.
It adds a coherent raw-junk execution, exact reversal and arbitrary-target clean
XOR workflow through ordinary `circuit.append(op)`.

The unchanged original eleven-cell notebook executes in an independent kernel.
The native thirteen-cell notebook has seven code cells and two SVGs. It executes
in a separate kernel blocking Qualtran, Cirq and Quimb imports. Both signature
and graph previews were visually inspected. Native names, tuple graph identities
and table layout replace foreign presentation objects. The source prose's
incorrect junk shape is explicitly distinguished from the implemented formula.

## Resource accounts

All rows below use the original default degree-eight polynomial `0x11d`.

| Account | Raw source-ordered inverse | Clean XOR inverse |
|---|---:|---:|
| Actual emitted CCX | 108 | 216 |
| Actual emitted CX | 920 | 1848 |
| Independent raw source model | 108 Toffoli, 920 Clifford | Twice raw plus 8 CX copy |
| Static original right-side bits | 40 | Separate wrapper contract |
| Caller-provided quantum workspace | 32 | 32 |
| Logical junk words retained | 3 | 0 |
| Measurements | 0 | 0 |

Raw junk is part of the output. Raw recursive scratch is zero. Clean execution
restores every workspace wire. Primitive counts use logical free output maps,
with no implicit fixed-wire binding SWAPs. Runtime samples and measurement costs
belong to their own accounts. Earlier accepted `0x11b` evidence retains its
216 CCX and 1894 CX clean count. Polynomial identity explains the different
linear gate total.

## Validation and remaining whole-file gaps

- 106 focused checks pass, including literal repaired-reference gates, all raw
  result/junk complex amplitudes with an entangled spectator at degrees 1–4,
  exact primitive reversal, and existing arbitrary-target clean XOR cleanup.
- 1852 concentrated field/resource checks and 96 independent-kernel/CPU/TN
  integration checks pass with no skips. Two existing TN warnings remain.
- Strict typing passes for all eight touched Python files. Black, Ruff, both
  ownership scopes, architecture boundaries, multilevel IR and public API
  migration checks pass.
- Five immutable source blobs match the frozen inventory. New numerical,
  declaration and example owners cover 101 statements and 34 branches at 98%
  combined coverage. Uncovered paths are an internal malformed-child guard and
  the mathematically unreachable zero-addition branch for admitted degrees.
- Accepted fixed-wire Kickmix inverse/erasure complex-branch and workspace
  evidence is reused through `test_kickmix_field_inverse`. This batch's raw
  source-chain and clean XOR execution are coherent, with zero measurements.

Q-0515 closes as a complete teaching asset. Q-0516 remains open for the
inherited `SynthesizeLRCircuit` adjoint defect and full generic
Program/foreign/automatic inverse/serialization admission. The reference remains
unchanged. Native exact linear inverses match the explicitly repaired reference
at gate level. Equality of resource counts does not close unchanged-source
mathematical gate parity. Existing source defect witnesses and accepted
measurement histories remain authoritative, with no duplicate capability IDs.

## Complete stored source schedule — 2026-11-20

`inverse_component.py` closes the concrete stored execution chain for the
complete eleven-method/property `GF2Inverse` source class. Constructor, two
factory, static-width, numerical, symbolic-model and notebook evidence above
is reused. The original builder reconstructs the identical native Program
through shared field-addition, Frobenius and Karatsuba child adapters. Exact
native inverse children retain the previously disclosed source defects.
No unchanged-source physical parity is claimed inside those defective children.

Input x remains THRU. Result and the original shaped junk array are RIGHT.
Degree two retains its empty junk array and source output aliases. Degree one
uses the original copy path. Physical slots may be freed and subsequently
reallocated, so ownership is verified sequentially against allocation events.
The final live set equals all returned words. Logical junk is retained data.
Only the original temporary and recursive polynomial padding are cleaned.

Complete native streams and aliases agree with the accepted exact physical
owner for degrees 1/2/3/4/5/6/8, including both degree-eight polynomials 283 and
285. All small full complex states through degree four retain source, inverse,
junk and an entangled spectator, with exact reversal restoration. Sparse full
complex states cover every input at degrees five, six and eight without dense
large-state allocation. Unused work is zero. The normal append test replaces
the previous public inverse emitter with a raising stub.

The degree-eight 285-polynomial raw graph emits 108 CCX and 920 CX, retains
three junk words and has 40 final logical bits. The 283-polynomial graph emits
108 CCX and 943 CX. Independent source Toffoli/Clifford estimates match these
categories. Measurements and classical SSA are zero. Detailed peak live storage
and returned aliases are in `g2-field-inverse-stored-evidence.json`. The existing
clean XOR wrapper and its doubled resource account remain separate.

Validation passes 24 focused checks, 2202 concentrated checks without skips,
17 local integration checks and strict typing for all three touched Python
files. Black, Ruff, team scope, architecture, multi-level IR and public API
checks pass. Six frozen source/dependency blobs match, Qualtran 0.7.0 and
Kickmix 0.1 Simulator are verified, and dependency policy reports 688 existing
repository findings with zero batch findings. Early test-only failures involved
a lifetime assertion that ignored slot reuse and passing a foreign QGF object
to the native polynomial codec. Both checks now follow the existing contracts.
Prior whole-repository failures remain recorded.

Q-0516 and Q-0517 remain open for the unchanged source adjoint defects and
generic foreign, symbolic physical, automatic inverse and serialization
admission. Module commit 47526bb29 owns the stored schedule; 06e4b820a owns
its normal append and complete source/complex-state verification.

```bash
python -m examples.algorithms.register_field_inverse
python -m pytest tests/unit/test_field_inverse_component.py -q
```

## Complete actual inverse-factory handoff

Both original `BloqExample` factories now have explicit foreign-boundary
consumers. The existing actual-source snapshot captures the concrete GF16
factory's entire forward graph using caller-owned clean RIGHT ports `result`
and `junk`. Its layout is `x[4], result[4], junk[4]`, with MSB words. Junk remains
a logical output whose value depends on the input. All twelve physically used
wires are logical ports, with zero private work and zero measurement events.

The captured forward stream has 104 CX, 18 CCX and 12 binding SWAP. An independent
flattening of the unchanged original graph verifies every primitive operand,
per-wire dependency and returned x/result/junk alias after tracking the physical
binding permutations. The original QEC model independently declares 104 Clifford
and 18 Toffoli. Its static QubitCount is twelve. Neither source model includes
physical binding SWAP or the retained external spectator.

Complete coherent states cover all sixteen input words entangled with a
spectator. All RIGHT words start clean. Both ordinary and reversed physical
binding execute through JSON and `circuit.append(op)` with Qualtran, Cirq,
Galois and Kickmix imports blocked during native execution. The complete complex
output matches an independent Cirq statevector execution of the original
flattened primitive graph and returned logical aliases.

The original forward recipe still differs from its mathematical
`on_classical_vals` callback because its direct linear-adjoint and Karatsuba
dependencies have the previously recorded source defects. The test keeps this
mismatch visible. It reuses the already accepted owned exact-adjoint and
mathematical inverse evidence without changing the reference or substituting
that corrected construction for the captured source graph.

The original concrete factory's generic backward capture fails before native
lowering in the same MeasureX quantum/classical arity mismatch independently
witnessed in the multiplication factory batch. Constructor serialization still
raises the original QGF TypeError. The symbolic factory rejects decomposition
and physical capture until its extent is concrete. These failure stages are
explicitly tested. The existing complete original 27-pass/3-skip/2-xfail suite,
whole-definition audit, stored inverse schedule, full-complex result/junk and
clean-owned-inverse evidence remain reused. Q-0516/Q-0517 stay open for their
original source, generic measured-inverse, codec and symbolic boundaries.

The concentrated inverse/Frobenius/linear/factory regression passes 275 cases
without skips in 7.66 seconds. Local integration passes 17 cases and the separate
Cirq 1.6.1 profile passes all three new actual-source consumers. Strict typing,
Black, Ruff, Ecosystem scope, architecture, multi-level IR and public API checks
pass. Two original source blobs and Qualtran 0.7.0/Kickmix 0.1 Simulator are
reverified. Separate resource and failure evidence is recorded in
`g2-field-inverse-foreign-evidence-20261121.json` and the corresponding focused,
regression, integration, Cirq and quality logs. No production code changes in
this follow-on batch.

## Complete inverse catalog dependency closure — 2026-11-22

The existing typed Itoh--Tsujii Program now supplies the concrete GF16 factory's
complete catalog decomposition and reachable dependencies. The root's ten
ordered calls, named operands and outputs match the original source. Recursive
walks retain addition, Frobenius, Karatsuba, linear synthesis, polynomial product,
constant multiplication and monomial dependencies. Across this root there are
13 exact source graphs, 40 original atomic/refusal contexts and two explicit
ignored-adjoint exceptions. Every comparable fresh source drawing agrees.

The two exceptions are existing native exact SynthesizeLRCircuit inverses, one
under the square adjoint and one under the Karatsuba dependency. The original
source emits its forward linear body for either adjoint flag. Native exact
inverse drawings are labeled `SynthesizeLRCircuit†`; source equality is not
claimed there. One nominal Join retains native finite-field identity over the
original QAny view. Width, side, physical calls and operands agree.

The presentation owner is `examples/algorithms/field_power_decomposition.py`.
It reuses existing Compiler Programs, whole-definition matching, product
presentations and Drawer glyphs. Source class labels require complete structural
matches and reject changed primitive operands. Numerical execution owners,
Core, Simulation and binding are unchanged from `4d0292339`.

Fresh atomic pre-T-decomposition measurement is 104 CX and 18 CCX, matching the
separate source QEC declaration of 104 Clifford and 18 Toffoli. There are zero
explicit T-family and measurement primitives. The separate analytical total-T
estimate is 72. Four allocation/free events retain their exact wire identities
and instruction offsets in the resource receipt. RIGHT result and junk are
logical outputs, while temporary recursive padding is released clean. The
existing complete complex-amplitude result/junk/spectator checks, append
execution, exact restoration and native Kickmix cases are rerun unchanged.

The shared square/inverse concentrated regression passes 439 checks without
skips or warnings. Seven frozen source/test/notebook assets are rehashed and
Qualtran 0.7.0 plus Kickmix 0.1 Simulator verified. The catalog adds 58 contextual
children across these two complete source groups, including 15 qualified
children. Thirteen have exact original source graphs and two are the explicitly
labeled native inverses. Existing erasure, generic dispatch, source arithmetic,
serialization and symbolic physical-execution boundaries above remain open.
Parent acceptance counts are unchanged.

The complete catalog regression passes 763 cases with three existing selection
exclusions and fourteen existing notebook escape warnings in 397.04 seconds.
Local integration passes 29 cases. Strict typing covers all five changed Python
files; Black, Ruff, integration scope, architecture, multi-level IR, public API
and whitespace checks pass. The final shared recursive observer passes 14 cases.
The concentrated 439-case execution/reference suite is rerun after observer
consolidation. Combined catalog inventory is 4104 JSON files, 4103 diagrams,
234 documents, 3750 contextual children, 84 decomposed roots, 269 atomic
interfaces and 427 qualified descendants. Three qualified descendants across
all field groups are explicitly labeled native exact linear inverses.
Receipts use `g2-field-power-ui-*-20261122` in the reference evidence directory.

## Complete original test closure — 2026-11-25

All six original test functions and their full slow/numeric parameter domains
are retained in the whole-file inventory. A new three-way count comparison
closes a previous evidence gap: for every degree 1–16, actual untouched source
decomposition counts equal source manual annotation and native retained child
identity/direction/multiplicity counts. Both original generalizers,
ignore_split_join and ignore_alloc_free, are applied exactly as in the original
test. This does not substitute annotation for actual decomposition or strip
nodes from an assertion that originally required them. Concrete original
GF16 factory counts also pass its own declared generalizers.

All ten raw inverse factory protocols are independently observed:

| Factory | Make | Decompose | Counts | Serialize | Strict types |
|---|---|---|---|---|---|
| gf16_inverse | PASS | PASS | PASS | FAIL | PASS |
| gf2_inverse_symbolic | PASS | NA | UNVERIFIED | FAIL | NA |

Both codec failures must identify QGF. The concrete native UI root equals the
existing field_inverse_program, retains every typed port and passes the shared
verifier. The symbolic source factory uses positive integer m. Its native domain
catalog accepts that exact symbol, retaining full signature and junk shape,
while the body remains unavailable. The original DocSpec deliberately exposes
only the concrete example, so the native UI index also omits the symbolic one.
The initial conformance test incorrectly looked it up in that index; using the
existing domain catalog corrects the test without expanding the UI catalog.

The physical-context Program codec refuses the concrete logical root. Existing
bound CircuitIR JSON, full original captured GF16 graph, coherent spectator,
reversed binding and raw logical junk evidence retain their separate roles.
Captured original forward emission is 104 CX, 18 CCX and twelve binding SWAPs,
with twelve logical wires and no private work or measurements. Counts and
physical-versus-logical output aliases remain separate resource accounts.
Mathematical callback/correct native inverse and the original circuit's
linear-adjoint defect are explicitly different witnesses. The source generic
backward capture still fails on MeasureX arity. No physical or mathematical
equivalence claim is inferred from count equality.

The unchanged complete square/inverse suites report 36 passes, six source skips
and five source xfails. Twenty independent raw protocols combine to nine PASS,
five FAIL, four NA and two UNVERIFIED. The new shared observer preserves raw
diagnostics and propagates unexpected errors; it replaces repeated protocol
orchestration in the addition test. Five original inverse lock blobs, including
the multiplication and square dependencies, retain identical values under
English repository paths. No reference file or production module changed.

These checks close Q-0517's actual original test responsibilities. Q-0516's
generic construction, automatic inverse, codec and source arithmetic/erasure
boundaries remain open. Receipts use `g2-square-inverse-test-closure-*-20261125`
and `g2-field-square-inverse-original-protocol-matrix-20261125.json`.

The final concentrated regression passes 315 cases, including the refactored
addition consumer, all complete square/inverse owner tests, actual foreign
capture, stored graphs, numerical execution and catalog export. Independent
Cirq 1.6 passes 172 cases. Strict mypy passes all four touched Python files;
Black, Ruff, Integration scope, language, local links, architecture, multi-level
IR and protected API checks pass. Production owners are unchanged, so the
accepted 145 local integration cases and unchanged 720 dependency baseline from
the preceding production batch are explicitly reused. Primitive counts remain
separate from the source resource model. All batch jobs are terminal.


## Raw degree-one inverse source branch — 2026-11-27

`inverse_decomposition.field_inverse_degree_one_program` implements the original
branch for any admitted concrete degree-one field word. Raw width may be greater
than one. The root allocates one same-type RIGHT result and calls the existing
literal degree-prefix addition. It returns x/result without a junk port,
measurements or private work. The actual power catalog selects this branch.

Unchanged GF(2), GF(3) and GF(5) roots match exact primitive calls, output leases,
complete source operators and every complex input/spectator amplitude. Clean
unused storage remains zero, and the captured native exact inverse restores
all physical input wires. Mathematical inversion and original scalar callbacks
remain independent: this degree-prefix raw circuit is not numerical GF(3)/GF(5)
inversion. Full higher-degree binary inverse and declared-adjoint catalog work
remain under Q-0516. The earlier source nonbinary higher-degree failure is not
generalized to this successful degree-one physical path.

The shared raw-field batch keeps Core/public contracts unchanged and uses normal
`circuit.append(op)` through the existing append adapter. Source/model evidence
is recorded in `g2-raw-field-resources-20261127.json` and the multiplication audit.


## Untouched-source clean-free violation intake — 2026-11-27

Before composing a literal binary inverse, an independent source-graph observer
now checks every original free point on all input basis words. It walks the
unchanged recursive decomposition down to CNOT/Toffoli leaves, assigns explicit
source allocation leases, and propagates exact Boolean masks for every input.
Since these leaves are phase-free permutations, a nonzero mask proves that the
released wire is not zero on those basis states and cannot be clean on arbitrary
coherent inputs. No numerical field callback or repaired source inverse is used.

| Source degree / polynomial | First violating primitive offset | Inputs with nonzero released work | Original measured primitive inventory |
|---|---:|---:|---|
| 4 / 0x13 | 65 | 12 / 16 | CX104, CCX18 |
| 8 / 0x11B | 552 | 250 / 256 | CX943, CCX108 |

The source reuses dirty released wires later in both examples. The probe does
not silently reset them at allocation or treat such reuse as valid clean work.
Degrees1/2/3 have no free points in the observed root; degree5's four recursive
padding frees are zero for every input. These results do not assert that the
source mathematical inverse value is correct merely because a release is clean.

`g2-original-inverse-free-intake-20261127.json` retains every offset, wire and
failing input word. The existing native mathematical inverse keeps its correct
square/constant adjoints and prior zero-work acceptance. Its literal-source
counterpart must explicitly preserve the static call evidence and refuse or
separately diagnose source-invalid clean releases; no clean-free contract may
be weakened to make it execute. Valid-source literal calls still need native
implementation. Q-0516 remains open pending that implementation and the complete
approved defect disposition. Count equality is not amplitude or cleanup evidence.

## Literal inverse dependency closure — 2026-11-27

`inverse_component._field_inverse_schedule` now owns one Itoh–Tsujii root
schedule. The existing mathematical `field_inverse_program` supplies exact
Frobenius and Karatsuba children. The new
`inverse_decomposition.field_inverse_decomposition_program` supplies literal
square and Karatsuba children, including their original declared adjoints.
Neither profile replaces or repairs the frozen reference. The degree-one raw
branch keeps its existing independent nonbinary storage behavior.

Inputs and outputs retain the existing MSB-first THRU x and RIGHT result/junk
interface. Logical junk is returned, not called clean work. Recursive padding
is private and checked at every free point. There are no measurement branches
in this forward source file. An exact inverse of the emitted native circuit
restores the physical input; it is distinct from the source's declared square
adjoint, whose original linear implementation ignores direction.

At degrees 2, 3, 5 and 9, the complete unchanged root with native literal child
adapters constructs exactly the same Program. A second observer descends the
untouched original dependencies to CNOT/Toffoli leaves and matches every
primitive operand and output alias. All input columns satisfy private releases
and leave unused work zero. Small degrees 2/3 additionally match the complete
original complex tensor with an entangled spectator via `circuit.append(op)`.
The mathematical profile's earlier complex, junk and clean-work checks rerun.

The executable literal constructor currently refuses every degree whose root
schedule frees the temporary word before later multiplication. Four/eight-bit
source violations are proven above. Refusing other unproved degrees is an
explicit coverage boundary, not an assertion that they all contain that bug.
A static original graph and the correct mathematical execution remain distinct
witnesses. The existing inverse catalog still selects the mathematical profile
for higher-degree roots; literal catalog dispatch and complete original-invalid
static graph qualification remain required before Q-0516 closes. No parent
acceptance is inferred from this dependency delivery.

## Literal static catalog and guarded append — 2026-11-27

The next dependency closure supersedes the preceding catalog gap.
`field_inverse_decomposition_graph` retains the full static original root,
including invalid clean frees, for source auditing and drawing. It does not
admit execution. `field_power_program` now selects that original graph instead
of the mathematical inverse. Whole-root verification maps square powers and
declared directions to child identities; generated names are opaque. Every
recursive GF16 source call and drawing now agrees exactly, with zero native
exact-inverse substitutions. Changing primitive operands rejects qualification,
and renaming all functions preserves labels.

The existing example `FieldInverse(..., literal=True)` uses the guarded literal
constructor through normal `circuit.append(op)`. The default remains the
mathematical inverse. Replacing the mathematical provider proves literal append
independent, while existing default append tests preserve its correct values.
The explicit literal profile fails before composition for unproved root releases.

An audit-only binding of the static four/eight-bit graphs matches the untouched
source stream exactly. All input columns reproduce the original 12/16 and
250/256 dirty-release violations and subsequent dirty reuse without any reset.
Those bound artifacts are used for diagnosis, not claimed clean executions.
Original scalar callbacks, source QEC declarations and primitive circuits remain
separate witnesses. No measurement branches occur in this forward root.

Q-0516 remains open: the conservative executable restriction on other root-free
profiles needs a complete valid-source disposition. Static construction is now
available for every concrete binary degree, but a static graph is not a proof
that its clean-release preconditions hold. Existing complete original tests,
models, callbacks, symbolic declarations and notebooks remain reusable. No
reference patch, generic clean-free waiver or parent acceptance is introduced.

## Complete source disposition with exact release analysis — 2026-11-27

This section supersedes the conservative degree-based refusal and the separate
static-graph helper described in the preceding historical batches. One literal
owner, `field_inverse_decomposition_program`, now retains the complete original
schedule at every concrete binary degree. The redundant graph wrapper is removed.
The existing mathematical `field_inverse_program` still supplies exact children.
No source graph, scalar callback or resource annotation is modified.

The append example binds the literal Program, then calls Simulation's shared
exact zero-release analysis on the actual CircuitIR and every clean free point.
The analysis consumes only Core types; Compiler neither imports Simulation nor
executes a numerical kernel. Its Boolean multilinear polynomials implement
XOR cancellation and idempotent Boolean multiplication, with a backward slice
from release points. A zero polynomial proves every input and entangled complex
superposition, rather than sampling a subset. A nonzero polynomial proves a
violating input exists and prevents composition. The same owner validates the
literal product's recursive padding. Provider replacement and unsupported-phase,
feedback, malformed-layout and expansion-budget cases fail before append.
See [the integration boundary](FIELD_CLEAN_RELEASE_ANALYSIS.md).

An independent untouched-source observer exhausts every input at degrees 1–16,
checking actual CNOT/Toffoli streams, frees and dirty reuse without resetting.
Native static streams, returned aliases and each exact free operand/offset match
the source at degrees 1–17. The symbolic proof agrees with the independent
truth-table result at 1–16 and proves every private release at degree 17.

| Source default degree | Complete all-input free result |
|---|---|
| 1, 2, 3, 5, 9 | All source frees zero (including no-free cases) |
| 4, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16 | Nonzero root temporary release and later dirty reuse |
| 17 | Exact Boolean proof of all private frees; no exponential truth table needed |

The default eight-bit source polynomial is 0x11D: 253/256 inputs violate its
root release. The separately retained 0x11B witness is 250/256. Neither result
is generalized to all polynomials or dimensions: execution admission evaluates
the actual circuit, with no degree whitelist. For example, degree 17 now follows
the same constructor and proof path as degree 5. Valid raw nonbinary degree-one
source circuits retain their separate prefix-addition evidence; original higher
nonbinary child failures remain refused rather than numerical field inversion.

All definitions in the complete source now have owners and evidence:

| Source responsibility | Current owner and evidence |
|---|---|
| Constructor/converter, signature, cached properties and degree-one omission | Existing Simulation field metadata, neutral typed signature/catalog and whole original factory protocols; source deprecation aliases are not new native public classes. |
| `_bits`, multiplier and `build_composite_bloq` | One shared root schedule, literal square/product children, complete recursive catalog identity/drawings and actual gate/alias/free-point comparisons. |
| `build_call_graph` and `my_static_costs` | Independent `inverse_model` and neutral signature width; complete degrees 1–16 and both original symbolic assumptions already verified. |
| `on_classical_vals` | Existing Simulation scalar inverse/junk callback, including zero and empty-array behavior; complete original numeric/slow tests and previous field sweeps reused. |
| Both factories, documentation specification and complete notebook | Existing whole-file inventory, actual raw protocols, frozen notebook execution and native ordinary append; only the concrete example belongs to the original DocSpec. |
| Inherited original transport/decomposition failures | Original QGF codec failures, symbolic physical refusal, generic measured SDK arity failure and invalid clean frees are retained, not reported as successful native source executions. |

The original whole suite (27 passes, three source skips, two source xfails),
notebook execution, scalar callbacks and symbolic models are unchanged evidence.
The actual native default mathematical inverse keeps its prior full complex,
retained-junk, zero-work and exact-reversal acceptance. Literal source execution
retains its own exact complex tensor checks at small degrees and phase-free
all-input/permutation checks at larger degrees. There are no measurement
branches in this forward source file. Source arithmetic defects are follow-up
issues under the explicitly approved faithful-migration rule; no correct inverse
claim is inferred from literal source parity or equal gate counts.

Analysis is explicitly bounded at 4096 relevant Boolean variables, 100000 live
monomials and five million symbolic monomial products. Unsupported gates or
exhaustion raise without admission. These internal verification cost limits are
separate performance boundaries; they do not discard source code, change the
input dtype, introduce a physical fallback or certify unproved zero work.
Measured primitive counts, source QEC estimates and static signature widths
remain separate resource accounts. No ReadTheDocs work, protected schema change,
CI wait, G3 work or merge is included.
