# Finite-field addition whole-file audit

G2-FILE-FIELD-ADD audits Q-0509 through Q-0514 and the direct converter dependency
Q-0524 against Qualtran v0.7.0 commit
`bcb17bb25360ed68ce3bc084645db3e929799389`. All seven original assets match their
frozen Git blobs. Q-0509, Q-0512 and Q-0524 close. Implementation parents Q-0510
and Q-0513 remain open for the source bookkeeping/count and generic admission
boundaries listed below. Test parents Q-0511 and Q-0514 close after the complete
actual factory/protocol mapping at the end of this audit. Existing source
failures remain reproducible.

## Ownership, interfaces and work

Compiler `finite_field/coefficients.py` remains the executable authority.
`gf2_add_ir(source, target)` preserves source and XORs it into target, in ascending
big-endian wire order. `gf2_add_constant_ir(target, value)` emits X on each one bit
of the big-endian constant, in ascending wire order. The existing controlled
extension retains its independently accepted behavior. The separate Kickmix
mixed-coefficient emitter retains its locked reversed traversal.

Compiler `addition_model.py` owns declared CX/X multiplicities and shape formulas.
It imports no numerical field library. Simulation `finite_field_addition.py`
owns conversion to the existing `FiniteFieldEncoding` and the two scalar field
callbacks. Dynamic galois polynomial/array classes remain inside Simulation.
Existing neutral register metadata represents signatures. Drawer and the native
catalogue present these declarations. There is no additional operation class,
IR or reference-package dependency in production.

The ordinary execution path remains `circuit.append(FieldOperation(n, gates))`,
using the existing example append adapter. Both operations require zero auxiliary
qubits and zero measurements. There is one coherent branch. Noncontiguous wire
binding preserves the logical big-endian order. Constant addition's inherited
source adjoint reverses its X stream and matches existing `Circuit.adjoint()`.
Register addition's source-specific adjoint returns the same operation. Repeating
the native register emitter preserves that canonical stream and restores state.

## Complete source definition disposition

| Original definitions | Native owner and disposition |
|---|---|
| `GF2Addition.qgf`, `__init__`, `bitsize`, `signature` | Existing `FiniteFieldEncoding`, `binary_field_encoding`, neutral `Register`/`Signature` and all example metadata. Native construction accepts the native field descriptor in place of a foreign QGF. The deprecated source `bitsize=` constructor/warning and attrs object construction are source API surfaces, recorded within the remaining generic admission boundary. |
| `GF2Addition.build_composite_bloq` | Existing ascending `gf2_add_ir` CX stream, with logical split/join represented by caller word binding. All widths 1 through 8 match unchanged source operands and order. Concrete words are required for execution. |
| `GF2Addition.build_call_graph` | `field_addition_calls` retains one CX edge of multiplicity degree. `field_addition_resource_formula` separates the signature width from physical instruction counts. |
| `GF2Addition.on_classical_vals` | Simulation `field_add_classical_values` checks both field scalar classes, preserves source identity and performs original field addition. All original domains at degrees 2 through 5 match. |
| `GF2Addition.adjoint` | The same self-inverse register emitter. Coherent append repetition restores the entire state, including spectator phase. |
| `_gf16_addition`, `_gf2_addition_symbolic`, `_GF2_ADDITION_DOC` | Native catalogue and complete notebook preserve concrete degree 4 and unrestricted symbolic m, parameter/register topics, signatures and declared graph. |
| `GF2AddK.bitsize`, `k`, `qgf`, `signature`, `_bits_k`, `is_symbolic` | Existing field codec, native catalogue and shared symbolic classification. All constants at degrees 1, 2, 3, 4, 5 and 8 match exact source X operands. Symbolic integer literals remain symbolic as in source. |
| `GF2AddK.build_composite_bloq` | Existing `gf2_add_constant_ir` provides exact concrete gate order and zero-constant empty emission. Native executable admission requires an exact Python integer fitting the word. Foreign numerical scalar construction remains a separate source API boundary. |
| `GF2AddK.build_call_graph` | `field_add_constant_calls` retains the zero X edge and the width bound for any symbolic width/constant. The known Python zero gives count zero, while SymPy zero follows the unchanged symbolic width bound. |
| `GF2AddK.on_classical_vals` | Simulation `field_add_constant_classical_value` preserves the source assertion and field constructor validation. All original degree 2 through 5 constants and inputs match. |
| Inherited `GF2AddK` adjoint | Existing circuit inverse preserves the exact reversed source X stream. No second inverse emitter is introduced. |
| `_gf16_add_k`, `_gf2_add_k_symbolic`, `_GF2_ADD_K_DOC` | Native catalogue and complete notebook preserve degree 4, k=1, and the source `positive=True, integers=True` symbolic assumptions. The source spelling is retained. |
| Entire direct `gf_utils.qgf_converter` | `binary_field_encoding` preserves native descriptor identity, integer/symbolic degree dispatch, polynomial identity and descending-power construction. Success and exception outcomes match, including accepted bool degree and empty polynomial powers. This closes Q-0524. |

The existing codec and source both support lazy numerical construction beyond
executable binary words. This batch preserves that boundary. General foreign
QGF/attrs admission and protected executable descriptor admission are not expanded.

## Complete original test disposition

There are five original functions in `gf2_addition_test.py` and four in
`gf2_add_k_test.py`. The native AST guard checks the whole sets. The unchanged
source suite reports **17 passes, six skips and six expected failures** in
2.64 seconds, with two original unregistered slow-marker warnings.

| Original test functions | Native assertion mapping |
|---|---|
| `test_gf16_addition`, `test_gf2_addition_symbolic` | Both entire factory signatures and model values match. Literal concrete source streams, symbolic costs, native SVG and independent notebook execution pass. Original generic autotester omissions remain below. |
| `test_gf2_addition_classical_sim_quick`, `test_gf2_addition_classical_sim` | Every original pair at degrees 2, 3, 4 and 5 matches both native scalar callbacks and XOR. Normal append independently checks the full complex state at widths 1, 2, 3 and 5. |
| `test_gf2_addition_resource` | Unrestricted symbolic degree preserves Clifford=m and T=0. Concrete streams contain exactly m CX gates. |
| `test_gf16_add_k`, `test_gf2_add_k_symbolic` | Both entire factory signatures, constants and declared calls match. All original symbolic assumptions and concrete/symbolic-zero dispatch are preserved. |
| `test_gf2_add_k_classical_sim_quick`, `test_gf2_add_k_classical_sim` | Every original constant and input at degrees 2, 3, 4 and 5 matches the native callback and XOR. Every constant's literal X stream also matches at degrees 1, 2, 3, 4, 5 and 8. |

Native coherent checks include one complex spectator, one initially zero unused
wire, two noncontiguous register words and repeated execution. Full complex
amplitudes agree within 1e-12. The unused wire remains zero. The implementation
introduces no measured branches, feedback, reset or auxiliary allocation.

## Complete notebook asset disposition

Each original notebook has 11 cells, including six code cells. All original
cells, documentation topics, two examples, graphical signature, one-level call
graph and leaf counts have native counterparts. Each native notebook adds an
append execution cell and explanatory Markdown, giving 13 cells and seven code
cells with two SVGs. Both unchanged originals execute in an isolated locked
kernel with a 60-second cell timeout. Kernel selection and timeout are test-only
environment configuration.

The native kernels block Qualtran, Cirq and Quimb imports. Both native notebooks
execute successfully, including normal append and inverse restoration. Native
signature table layout, operation names and resource labels replace the source
drawing layout. Actual native DOT renders were inspected as PNGs. This explicit
presentation adaptation closes Q-0509 and Q-0512.

## Separate resource accounts

| Operation | Actual emitted primitives | Source declared model | Width and workspace |
|---|---|---|---|
| Eight-bit register addition | CX 8 | Clifford 8, T 0, measurement 0 | 16 logical operand wires, auxiliary 0 |
| Eight-bit constant 0xA5 | X 4 | Clifford 4, T 0, measurement 0 | 8 logical operand wires, auxiliary 0 |
| Eight-bit Python constant zero | Empty stream | Explicit X edge with multiplicity 0 | 8 operand wires, auxiliary 0 |
| Symbolic m/k constant | No executable word emitted | X bound m | Signature width m, auxiliary 0 |
| Four-bit full source register decomposition | CX 4 plus two Split and two Join nodes | Declares only CX 4 | Bookkeeping nodes emit no physical gates |
| Four-bit full source constant-one decomposition | X 1 plus one Split and one Join node | Declares only X 1 | Bookkeeping nodes emit no physical gates |

The formula's logical width is the operand signature width. It is separate from
recursive graph peak estimates and runtime statistics.

## Central remaining boundaries

| Parents | Reproduced boundary | Owner and closure condition |
|---|---|---|
| Q-0510, Q-0513 | The two concrete source autotester count checks fail because annotations omit QGF Split/Join nodes. Physical X/CX counts match. Source diagnostic compares the entire declaration and decomposition without editing either. | Compiler resource owner. Complete native stored graphs preserve the bookkeeping identities. Original annotation disagreement remains a source defect; test-parent closure does not close these implementation parents. |
| Q-0510, Q-0513 | Source attrs/QGF construction, deprecated `bitsize=` warning and generic foreign object admission are not native production constructors. Explicit native field resolution and supported concrete words are accepted. | Existing Core metadata, Simulation conversion and Ecosystem boundary. Establish the remaining native admission/failure mapping without changing protected schemas or introducing a parallel Bloq API. |
| Q-0511, Q-0514, closed | Six source fixture skips cover symbolic decomposition/count applicability and dtype checks. Four serialization xfails explicitly reject QGF. The two other xfails are the counts diagnostic above. | The new actual-factory conformance test independently invokes all twenty original protocols and checks native signatures, stored owner identity and verification, symbolic refusal, and logical-versus-physical codec rejection. Bound CircuitIR JSON and numerical execution remain separate evidence; no successful source QGF codec is claimed. |

Q-0516/Q-0519 historical linear-adjoint and Karatsuba issues are unchanged. This
batch reuses their accepted dependency evidence and does not reopen them.

## Batch verification

- 43 targeted whole-file assertions pass.
- 1,604 concentrated field, polynomial, classical-field and resource tests pass
  in 17.98 seconds, with zero skips.
- 90 independent-kernel, CPU and tensor integration checks pass in 19.86 seconds,
  with zero skips and two existing tensor API deprecation warnings.
- Strict typing passes for all five changed Python files. Black, Ruff, both
  domain scope checks, architecture, multi-level IR and protected API checks pass.
- The three new production/example modules have 68 statements and 14 branches,
  all covered. Existing protected API and schema checks pass.

## Complete actual addition-factory consumers

The complete `gf2_addition.py` and `gf2_add_k.py` factory sets are now consumed
through the existing actual-source named emitter. Each original module has one
GF16 concrete example and one unresolved symbolic example. Their complete
source-definition, original tests, numerical callbacks, typed native Program and
tutorial evidence above is reused.

Both directions of the actual GF16 addition emit four CX, with zero workspace,
feedback and binding SWAP. The actual constant-one factory emits one X in either
direction with the same zero overhead. Logical source QubitCount is eight for
addition and four for constant addition. Original QEC declarations independently
match the physical Clifford counts. No measurement branches occur.

Every concrete input word, arbitrary coherent target word and spectator is
covered in a full complex state, under both ordinary and reversed physical
binding. Every primitive operand and per-wire order agrees with an independent
flattening of the unchanged graph. Captured IR round-trips JSON and executes
through `circuit.append(op)` while all optional source SDK imports are blocked.
Both original directional streams restore the complete coherent input when
composed. GF2AddK uses the original generic adjoint, whose classical callback is
NotImplemented; its independently checked self-inverse forward action supplies
the expected numerical result, with the original adjoint graph checked separately.

All four actual factory constructors reproduce the original QGF serialization
TypeError. Both unresolved examples reject physical decomposition and capture,
while their symbolic Clifford bounds remain independently checked. The original
GF2Addition deprecated `bitsize=` warning and equality with the canonical factory
are verified. Source bookkeeping-count omissions and the original remaining
generic/autotester boundaries stay explicit. These consumers do not change the
original serializer or claim that its QGF codec succeeds.

The combined addition/Frobenius/direct-dependency regression passes 356 cases
without skips in 5.55 seconds. Local integration passes 17 cases and independent
Cirq 1.6.1 passes all nine new factory consumers. Strict typing, Black, Ruff,
Ecosystem scope, architecture, multi-level IR and API checks pass. Six unchanged
source/test blobs and Qualtran 0.7.0/Kickmix 0.1 Simulator are reverified. Separate
primitive and model rows are in `g2-field-linear-foreign-evidence-20261121.json`.

## Complete stored catalog graphs and direct dependencies — 2026-11-22

The complete gf2_addition.py and gf2_add_k.py catalog groups now use the existing
Compiler field_addition_program. Concrete factory graphs retain exact ordered
Split/CNOT/Join and Split/X/Join calls, operands, field identities and freshly
computed source drawings. The four original factories, DocSpecs, filenames and
symbol assumptions are retained. The two symbolic factories keep their original
DecomposeTypeError. Existing CNOT, X and typed Split/Join symbols are reused.

The UI graph publishes eight contextual dependency nodes under the two concrete
roots. These are original atomic/refusal children, with native interface
presentation. They introduce no additional decomposable source-class labels.
All 22 field/polynomial factories retain their original metadata. The duplicate
constant-multiplier factories still share their original filename. Independent
whole-field intake records successful omitted roots for subsequent complete
multiplication, Frobenius, inverse and polynomial batches.

The exact catalog Programs equal the existing executable owner objects. The
Compiler, Core, Simulation, binding and accepted execution-test files are
byte-identical to published 2203ce443. Fresh verification reruns 282 concentrated
cases, including all source constants through eight bits, original classical
domains, complete complex vectors and entangled spectators, noncontiguous MSB
bindings, returned aliases, zero-work/zero-measurement paths and normal append
restoration. The single coherent branch is the complete measurement-branch
scope. Twenty source graph/drawing cases cover widths 1, 2, 3 and 8 and zero,
all-ones and sparse constants. Twenty-nine local integration cases pass.

Actual catalog GF(16) register addition emits four CX, while constant one emits
one X. The independent source model reports four and one Clifford respectively.
Both streams have zero T, measurement and workspace events. Full primitive
histograms and analytical model fields remain separately recorded in
ftqc-reference-env/g2-field-addition-ui-source-intake-20261122.json. The original
source bookkeeping/count, generic construction and serialization boundaries
above remain open under Q-0510/Q-0513. Test parents Q-0511/Q-0514 close only after
the independent protocol mapping below, rather than catalog graph acceptance
alone.

Strict mypy covers four Python files. Black, Ruff, integration scope,
architecture, multilevel IR, protected API and whitespace checks pass. Seven
frozen source/test/notebook/converter blobs match their ledger identities.
Locked Qualtran 0.7.0 and Kickmix 0.1 Simulator are verified. The initial new
trace harness attempted an integer-only reference constructor on a field port.
The corrected harness reuses its static source identity observer directly,
preserving the original QGF descriptor. The published child inventory includes
all four independently named rebinding definitions in register addition and
both rebinding definitions in constant addition, plus CNOT and X.

## Complete original test closure and actual protocol mapping — 2026-11-25

Both entire original test files are reread and their frozen blobs reverified.
All nine top-level functions retain their existing AST inventory guard. The
unchanged full suite, including every slow degree-three/four/five case, reports
17 passes, six original skips and six original xfails. Existing exhaustive
classical callbacks, all constant streams, exact source operands and stored
Split/CNOT-or-X/Join graphs are reused, including source-defined adjoints.

The actual four factory examples now run every original raw protocol
independently in both the source receipt and the native conformance test:

| Actual factory | Make | Decompose | Counts | Serialize | Strict types |
|---|---|---|---|---|---|
| gf16_addition | PASS | PASS | FAIL | FAIL | PASS |
| gf2_addition_symbolic | PASS | NA | UNVERIFIED | FAIL | NA |
| gf16_add_k | PASS | PASS | FAIL | FAIL | PASS |
| gf2_add_k_symbolic | PASS | NA | UNVERIFIED | FAIL | NA |

All twenty observations comprise eight PASS, six FAIL, four NA and two
UNVERIFIED. Count failures must name both omitted Split and Join; serialization
failures must name QGF. None is filtered or repaired. Symbolic type checks are
inapplicable because decomposition is unavailable. Native construction checks
the actual catalog signatures, shapes, sides and descriptors. Concrete roots
equal the existing executable field_addition_program objects and pass the
shared verifier; both symbolic roots deliberately have no body.

The source field degree is not interchangeable with its unresolved signature
width. Both native and source signatures retain ceil(log2(floor(2**m))) while
their declared CX/X bound uses degree m. The native descriptor retains m as its
degree parameter. The first conformance run incorrectly compared these two
different source properties; the corrected assertion checks degree against
degree and the complete signature against the original complete signature.
No symbolic expression or original assumption was changed to simplify it.

Native physical-context Program serialization refuses these logical graphs.
That expected boundary is now asserted on both actual concrete catalog roots,
while existing bound CircuitIR JSON tests cover the separate executable
representation and full complex-state execution. Four source QGF codecs remain
failed; no new generic logical Program codec or protected schema is introduced.
The original polynomial identity is checked on each actual concrete root.

The complete callback, stored graph, native catalog, bound append and independent
foreign-capture consumers remain separate owners. Existing full complex vectors
include spectators, arbitrary target words, forward/inverse restoration and
noncontiguous bindings, with zero work and zero measurement events. The single
coherent branch is the complete measurement scope. Concrete GF16 emits four CX
or one X, versus independent model four or one Clifford. Symbolic gate bounds
are models only, not emitted primitives.

These observations close Q-0511/Q-0514's complete original test responsibilities.
They do not close Q-0510/Q-0513's remaining generic construction and codec
interfaces. Receipts use `g2-field-addition-*-20261125`, including the independent
twenty-row source protocol matrix and untouched original full-test log. Final
regression and strict-check results are recorded in the synchronized ledger.

The accepted regression passes 122 concentrated cases; after replacing the
old CSV-dependent lock fixture, the entire 47-case owner file passes again.
Cirq 1.6 independently passes 66 owner, typed-binding and actual-factory cases.
Strict mypy passes the complete touched Python file. Black, Ruff, Integration
scope, language, local links, architecture, multi-level IR and protected API
checks pass. All production modules are unchanged, so the accepted 145 local
integration cases and 720 dependency-baseline findings from the preceding batch
are reused explicitly. The final lock fixture checks exactly the same seven
original blob values through English paths, without retaining Chinese CSV keys
inside repository code. No reference data was translated or changed.

An initial quality run could not write its architecture receipt because the
volume was temporarily full. After confirming there were no live pytest jobs,
only completed pytest temporary directories were removed. The full strict-check
sequence then completed successfully. The source checkout, dependency packages,
published evidence and working changes were retained.

The complete catalog regression passes 749 cases with three existing
deselections and 14 original notebook warnings. Every combined file equals its
independently generated domain export, and every contextual child link resolves.
Combined totals are derived from the preceding published export plus eight
exact new field children. They are 3833 JSON files, 3832 diagrams, 234 docs,
3479 contextual children, 75 decomposed roots, 278 atomic interfaces and
345 qualified decomposable descendants. Q-0120 remains open for broader groups.
