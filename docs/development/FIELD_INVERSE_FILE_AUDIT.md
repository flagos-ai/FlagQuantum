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
