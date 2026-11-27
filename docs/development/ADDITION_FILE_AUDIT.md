# Addition whole-file audit

G2-FILE-ADDITION audits Q-0173, Q-0174 and Q-0175 from Qualtran v0.7.0 commit
`bcb17bb25360ed68ce3bc084645db3e929799389`. Q-0173 closes as a complete notebook
asset adaptation. Q-0174 and Q-0175 remain open for the specific boundaries below.
Existing arithmetic emission and numerical execution are reused unchanged.

## Ownership and interfaces

Compiler owns addition structure, literal ordered primitive emission, typed
component graphs, resource identities and register-lifetime estimates. Simulation
owns numerical values and tensor contraction. Core owns existing metadata only.
Drawer owns glyphs and rendering. Examples compose these interfaces and contain
no reference imports. The user continues to use `circuit.append(op)`.

All executable words are big-endian. Add preserves its left word and updates
its right word modulo the latter's width. The emitter owns `right_width - 1`
clean carries and distinct measurement slots. AddK additionally owns a loaded
constant word, unloaded before release. An out-of-place sum owns its output
word, with an optional most significant carry bit. Its adjoint consumes a
correct sum and clears it through measured AND erasure. No new quantum work,
measurements or execution kernels are introduced by this presentation batch.

## Complete source definition disposition

The implementation has three classes, seven declared fields, 31 methods,
ten example factories, one unused private converter and three documentation
specifications. There is no separate export list. The native copyright headers
retain source attribution. The frozen implementation, tests and notebook are
verified against the original inventory's Git blobs.

| Source definitions | Native owner or asset disposition | Evidence and remaining scope |
|---|---|---|
| `Add.a_dtype`, `b_dtype`, `b_dtype_default`, `_a_dtype_validate`, `_b_dtype_validate`, `dtype`, `signature` | `ArithmeticCall`, existing Core register types/signatures and explicit emitter word admission | All ten example metadata rows match. Resource encodings retain signed, unsigned and Montgomery identity. Native concrete word admission is separate from source attrs and foreign Cirq construction. General dtype-specific admission/cache remains open. |
| `Add.decompose_bloq`, `_left_building_block`, `_right_building_block`, `decompose_from_registers` | `addition_structure`, `addition`, `addition_component` and shared register binding | Accepted forward and exact adjoint source sequences, operands, feedback, coherent amplitudes, exhaustive small branches, clean carries and append execution are reused. |
| `Add.on_classical_vals` | Simulation `register_add_classical_values` | Original integer conversion, wrap, mixed signed words and 32-bit examples are checked against source. |
| `Add._circuit_diagram_info_`, `wire_symbol` | Drawer `addition_symbols` through the existing score callback | Frozen bit labels and every register side match. Add labels are `a` and `a+b`, with an empty connector. |
| `Add.build_call_graph` | `arithmetic_declarations`, QEC query and resource-width owner | Full concrete/symbolic costs, child identity, order and width-cache evidence are reused. Source one-bit and unequal-width model discrepancies remain explicit. |
| `Add.get_ctrl_system` | Existing controlled addition and zero-control wrapper | Both polarities, exact inverse, source operands/feedback and coherent branches are already accepted. Original 32-bit classical conversion assertions are added. |
| `OutOfPlaceAdder.bitsize`, `is_adjoint`, `include_most_significant_bit`, `out_bitsize`, `signature` | Existing arithmetic identity and `out_of_place_addition` | Forward/right output and inverse/left consumption, full/truncated output, concrete and symbolic models already accepted. All three example signatures match. |
| `OutOfPlaceAdder.registers`, `apply`, `with_registers` | Source Cirq ArithmeticGate admission boundary | `registers` returns binary radix lists. `apply` adds unbounded `c+a+b`. `with_registers` raises NotImplementedError. Native execution uses explicit words and Simulation callbacks. A generic foreign ArithmeticGate protocol is still outside accepted admission. |
| `OutOfPlaceAdder.adjoint`, `__pow__` | Existing explicit direction and exact measured inverse | Explicit source adjoint preserves truncation. Source power `-1` drops the truncation flag. That witness is reused and the file stays open. |
| `OutOfPlaceAdder.on_classical_vals` | Simulation `out_of_place_sum_classical_values` | All original one-through-four-bit pairs pass in both directions. Source inverse discards even an incorrect sum, independently of physical cleanup acceptance. |
| `OutOfPlaceAdder.decompose_from_registers`, `build_call_graph` | Shared `out_of_place_addition_structure`, emitter, declarations and width owner | Both flags/directions have literal source streams, complex branch and clean-output evidence. Source full-width prose versus loop/model discrepancy stays recorded. |
| `OutOfPlaceAdder.wire_symbol` | Drawer addition callback | Connector is `c=a+b`. Through `a` displays the source inherited `OutOfPlaceAdder` label. Other canonical ports preserve their labels/directions. |
| `AddK.dtype`, `k`, `__attrs_post_init__`, `signature` | Existing arithmetic identity, encoding declarations and Core metadata | The three original constant examples preserve widths, signedness and constants, including symbolic n/k and signed 64-bit -23. Foreign attrs construction remains an admission boundary. |
| `AddK.on_classical_vals` | Simulation `add_constant_classical_value` | Existing conversion, wrap and symbolic rejection tests are reused. |
| `AddK._load_k_bloq`, `build_composite_bloq` | Existing constant load/Add/unload emitter and component bindings | Source negative unsigned normalization, exact inverse rather than adding -k, controlled loads and clean constant/carry ownership are already accepted. |
| `AddK.build_call_graph` | Shared arithmetic declarations, QEC and width models | Original XorK twice and Always(Add) once match, including symbolic constants and both resource profiles. |
| `_cvs_converter` | Unused source helper, explicit retirement | AST/search confirms no use in this file. Existing control-register codecs own conversion. No duplicate helper is introduced. |
| Ten `_add*` factories | `addition_catalog.addition_examples` | Every instance's signature, dtype, constant and one-level graph matches the unchanged factory. |
| `_ADD_DOC`, `_ADD_OOP_DOC`, `_ADD_K_DOC` | Native notebook teaching text | All parameter/register/example/model topics are covered, with source discrepancies identified explicitly. Native callable names replace the upstream documentation generator. |

## Complete original test matrix

There are 25 original test functions plus `add_reference_t_complexity`. The new
AST guard checks this complete set. Source-only execution is distinct from native
assertion evidence. The unchanged source run gives 267 passes, nine skips and
five expected failures, excluding its notebook. The original `test_notebook`
also passes unchanged in a separately configured locked Jupyter kernel.

| Original function | Original parameter scope | Native assertion evidence |
|---|---|---|
| `test_examples` | Ten factories across five upstream autotester modes | All ten native metadata/one-level declaration graphs and notebook examples checked. Original fixture skips/expected failures below remain open. |
| `test_add_decomposition` | a,b in 0..3, widths 3/4 | Existing literal source stream, all-small coherent branch and work-cleanup tests. New original bit labels cover both widths. |
| `test_add_diff_size_registers` | a 1/2, b 1/2/3, widths 2/3 and 4/5 | Existing unequal-width source stream and coherent amplitude evidence. Every original width pair's bit labels are added. |
| `test_add_truncated` | 4+4 at 3/4 bits and 4+7 at 3 bits | Existing modulo-output full-state addition and constant-add tests cover overflow and clean carries. |
| `test_subtract` | a,b in 0..3, widths 3/4 | Existing signed classical and full complex source-gate evidence, interpreting Add(a,-b) as the original test does. |
| `test_addition_gate_counts` and helper | Widths 3..9 | Existing arithmetic declarations, historical budgets and separate literal primitive counts. |
| `test_add_tensor_contract` | All 64 a,b pairs 0..7 in five-bit words | New complete native 1024-by-1024 matrix equals source. All original entries are asserted from the cached matrix. |
| `test_add_call_classically` | a,b in 0..3, widths 3/4 | Existing legal-domain classical action and conversion tests. |
| `test_add_call_classically_overflow` | 5+6 modulo 8 | Existing exhaustive three-bit classical action tests. |
| `test_add_truth_table` | Entire formatted two-bit table | Existing native truth-table enumeration, formatting and measured graph-consumer tests. |
| `test_add_in_cbloq` | Four-bit composition and cost | Existing typed addition graph verification, binding, tensor providers and cost queries. |
| `test_add_classical` | Signed 32-bit 10+3 | New native/source decomposed callback equality. |
| `test_add_symb` | Signature 2n and peak Max(3,2n) | Existing full symbolic width cache/model and new symbolic example signature/graph check. |
| `test_add_mixed_signed_result_reg` | All unsigned4/signed4 pairs | New exact original domain through native classical action. Existing gate evidence is reused. |
| `test_add_mixed_signed_op_reg` | All signed4/unsigned4 pairs | New exact original domain through native classical action. |
| `test_out_of_place_adder` | All three-bit pairs and inverse maps, model T12/0 | Existing source streams, all coherent input amplitudes and every small erasure history, plus original callback domains below. |
| `test_controlled_add_k` | Symbolic zero-control graph | Existing controlled-constant resource declarations retain positive child and X twice. |
| `test_add_k_decomp_unsigned` | Width5, k5/8, no control or 01/10/11 | Existing ordinary/grouped controlled constant-add source-stream, work and resource tests. |
| `test_add_k_decomp_signed` | Width5, k-5/8, same controls | Existing signed encodings, loading and shared controlled source-stream evidence. |
| `test_classical_add_k_unsigned` | Four original control/value tuples | Existing native constant action and grouped-control execution evidence. Exact generic original callback/autotester replacement remains within the open test boundary. |
| `test_classical_add_signed_overflow` | Widths2..4, maximum+maximum gives -2 | Existing signed modulo/classical and coherent source-gate evidence. |
| `test_classical_add_k_signed` | Width5 k2 with control10 and width6 k-3 | Existing signed constant action and controlled emission evidence. Exact generic original callback replacement remains open. |
| `test_notebook` | Entire addition notebook | Original function passes with isolated locked kernel. Entire native notebook runs independently and produces six SVGs and three PNGs. |
| `test_outofplaceadder_classical_action` | All pairs at widths1..4, both directions | New native callbacks match every original source result. |
| `test_controlled_add_from_add` | Width32, active/inactive zero control | New native callback equality for both original inputs, reusing existing specialized-control source stream evidence. |

Native tests retain explicit ownership instead of implementing another source
Bloq class or a generic reflection adapter. Therefore Q-0175 remains open until
its original autotester/callback boundary receives an accepted native mapping
and the inherited source fixture failures have a closure disposition under the
frozen criteria. Source expected failures are recorded as failures of that
assertion path rather than counted as successful native verification.

## Complete notebook asset disposition

Q-0173 has 33 cells, including 20 code cells. Its missing file was restored from
Git blob `c2a3acbc0f0c6509f9b05bd760d30c3661d499d7`, verified without modifying the
reference. The native `register_addition.ipynb` preserves cell type/order and
all ten examples. Symbolic declarations use existing Compiler resource graphs.
Concrete/symbolic signature tables use Core metadata and Drawer table rendering.
Three concrete musical-score examples use the same accepted glyph callback.
The runnable append path links to `classical_register_addition`.

Native resource names and table layout are explicit presentation adaptations.
Tables show directions, encodings and widths even for symbolic metadata. The
source musical-score signature layout is replaced by the existing native table
renderer, with three additional concrete score views. Source model counts are
shown as declarations. The full-width out-of-place source prose claims 4n-4T,
while its loop/model costs 4nT. The teaching text states both cases explicitly.
This documentation correction leaves the frozen code and model unchanged.

## Reused implementation evidence

The complete accepted Add/AddK/OutOfPlace trajectories and source-gate evidence
include `dd724f6b`, `ca3a5b9a`, `f5af8f5a`, `60c4be94`, `e2105c20`, `7789c356`,
`4042c98e`, `511247d0`, `9e2329ad`, `305dccd4`, `35d00a3a`, `b1519f5f`, plus
recent automatic-width and exact-adjoint batches. Their tests remain in the
concentrated regression. Existing full complex inputs, entangled spectators,
small exhaustive measurement histories and zero-work restoration are reused.

## Separate resource accounts

These are eight-bit forward examples. A T entry includes T and TDG together.
Measurement-conditioned CZ and reset are listed explicitly. Models use shared
native declarations and match their source profiles.

| Operation | Measured primitives | Modern source model | Historical source model |
|---|---|---|---|
| Add | CX81, H21, T/TDG28, S7, measurement7, conditional CZ7, reset7 | AND7, Clifford46, measurement7 | T28, Clifford130, rotations0 |
| Truncated out-of-place | CX79, H14, T/TDG28, S7 | AND7, Clifford37, measurement0 | T28, Clifford100, rotations0 |
| Full out-of-place | CX88, H16, T/TDG32, S8 | AND8, Clifford40, measurement0 | T32, Clifford112, rotations0 |

Presentation and metadata paths add zero primitives, quantum work and
measurements. Historical Clifford accounting is a source declaration profile.
It is kept separate from the sum of measured primitive opcode counts.

## Centralized remaining boundaries

| Parents | Evidence and impact | Owner and closure condition |
|---|---|---|
| Q-0174 | Add one-bit negative declared Clifford/CNOT counts and unequal-width declared-count disagreement, already witnessed by earlier source-model batches | Compiler resource owner. Preserve frozen diagnostics and physical/model separation. Parent closure needs explicit disposition under existing criteria. |
| Q-0174 | Out-of-place source `**-1` drops truncation while explicit `.adjoint()` preserves it. Source prose and full-loop T costs disagree. | Compiler exact-adjoint and documentation owners. Preserve original flag witness and distinguish corrected native documentation from unchanged source. |
| Q-0174 | Generic source attrs/Cirq ArithmeticGate admission, dtype-specific cache seeds and foreign register rebinding lack a complete native contract | Existing metadata, compiler and ecosystem boundaries. Audit supported constructor/protocol failures without expanding protected Core schemas or adding foreign reflection. |
| Q-0175 | Original examples have nine fixture skips for symbolic decomposition/count/dtype checks and five counts expected failures | Test/resource owners. Affected counts examples are unequal-width Add, concrete out-of-place 4/64 and concrete AddK 4/64. Retain original results and add an accepted full native autotester/callback mapping before closure. |
| Q-0175 | Generic original controlled-constant callback assertions are represented by accepted emitter/action evidence, not an exact native replacement of the original autotester interface | Existing Simulation/component evaluators. Complete those original callback cases through the normal native graph entry, without another compatibility Bloq. |

Q-0173 closes. Q-0174 and Q-0175 remain open with these specific conditions.
Further unchanged source-defect reruns are unnecessary. The next whole-file
batch should select a finite-field source/test pair and reuse its existing
accepted emitters and Simulation callbacks rather than extend Add scenario IDs.

## Validation

The focused addition asset/glyph tests pass 35 cases. The final concentrated addition
family regression passes 4,580 tests in 84.71 seconds without skips. Independent
Jupyter-kernel, CPU vertical-path and tensor-network integration passes 88 tests
in 23.90 seconds without skips. Strict typing covers five changed Python
files. Black, Ruff, scope, architecture, internal IR and protected API checks
pass. The normal append example preserves source/control and clears all three
work wires while reporting primitives separately from its model. Source and native notebooks execute all cells. Native SVGs and
three score images are rendered, and the score collage is visually inspected.

Initial glyph comparison exposed inherited Cirq first-port labeling. The callback
was corrected against the unmodified source. A first headless notebook run
selected an unavailable Tk backend, corrected to Agg in the test only. A glyph
Literal type inference error and missing notebook cell IDs were also corrected.
These setup diagnostics are separate from the retained source fixture outcomes.

Drawer implementation and source glyph assertions are committed as `4318c0e4`.
The complete native catalog, notebook and original assertion audit are committed
as `d7cf6d32`. The final batch reuses unchanged quantum emission and numerical
owners. The image review exposed a long-label overlap in the first preview.
Canonical display titles and wider notebook spacing corrected it before final
integration. Source notebook execution uses a test-only kernel selection and
60-second timeout in place of the source runner's default environment/600 seconds.
The original test function and all notebook code cells remain unchanged.

## Native component graph completion — 2026-11-20

The complete executable Add/AddK/OutOfPlaceAdder source-file batch now shares
existing typed Add, bitwise, Always, And, control and lifetime owners. Add's
accepted forward and exact-adjoint evidence is reused. No reference source is
modified. The extraction of bitwise control propagation has two concrete
consumers, conditional complement and controlled constant loading.

`constant_addition_program(dtype, pattern, adjoint=False, control_value=None)`
accepts an encoded unsigned pattern in MSB-first storage. Numerical signed-value
encoding belongs to the caller. It retains allocation, XorK, Always(Add), XorK
and release. The exact adjoint retains the same load/unload and uses Add-adjoint.
Positive control acts directly on the load calls. Negative control retains the
original X/positive-body/X wrapper. Always(Add), allocation and release preserve
the through control without controlling their bodies. The loaded constant and
all Add carries are clean on return.

`out_of_place_addition_program(width, include_msb=True, adjoint=False)` retains
unsigned a/b and creates a RIGHT c, or consumes a LEFT c in the inverse. The
forward graph owns one allocated low output and the And-created higher outputs.
Its adjoint consumes those And outputs by measurement and frees the low bit.
There is no independent scratch pool beyond c. A bound forward graph explicitly
transfers its live outputs to the caller. A separately bound inverse requires
explicit pool ownership of the correct sum. Incorrect c is outside physical
cleanup admission, although the source classical callback discards it.

Source Cirq import schedules independent XORs ahead of Allocate. Tests compare
canonical dependency edges and named operands after instance-ID renaming, as
well as the complete primitive stream and feedback. At width one, source Cirq
omits Split/Join views. Native typed integer ports retain these zero-cost views.
Their physical gates, operands, output order and ownership still agree. These
view occurrences are not charged as primitive gates.

Both append consumers execute the native graph. The out-of-place example checks
coherent sum amplitudes and sampled erasure histories. Native conformance checks
all small raw erasure histories, unnormalized complex amplitudes, an entangled
spectator and clean work. AddK checks every forward/inverse history pair, three
integer descriptors, both control polarities and both directions. Wider cases
compare physical streams, feedback and resource counts without dense simulation.
The actual concrete source factories are AddK 4-bit k=2 and signed 64-bit k=-23,
and OutOfPlaceAdder 4 and 64 bits. They use all five existing native check modes.

Measured examples remain distinct from analytical categories. Three-bit k=-3
emits 4X, 6H, 4T, 4Tdg, 21CX, 2S, 2measure, 2CZ and 2reset. Its source model is
2And, 15Clifford, 2measurement and peak8. Two-bit full sum emits 22CX, 4H, 4T,
4Tdg and 2S, while its source model is 2And, 10Clifford and peak7. Its inverse
emits 10CX, 2H, 2measure, 2CZ and 2reset, while the independent inverse model
is 12Clifford, 2measurement and peak7. Other model categories are zero.

Q-0174/Q-0175 remain open for symbolic physical/generic foreign-object and
original autotester/classical callback admission, with original power-1
truncation loss and source estimate discrepancies recorded above. This native
execution graph completion does not certify those wider boundaries. Proposal108
remains pending for wider classical Program admission. Whole-file source test,
notebook and numerical callback evidence is reused.

Final local acceptance has 372 focused and 3,885 concentrated passes, with two
existing notebook/environment warnings and no skips in those runs. Strict typing
covers all nine changed Python files. Black, Ruff, team scope, architecture, IR,
public API validation, both append examples and four related integration checks
pass. The default suite has 434 passes and 51 skips before the established
`test_every_module_with_examples_is_covered` failure. Its reported nine modules
and failing test are byte-identical to this batch's base `e59ec616a`.
The dependency policy reports 688 whole-repository findings and none in the
changed paths. Local evidence receipts use the `g2-addition-native-*` prefix.

## Complete foreign consumer and factory closure

The final closure covers all 45 recursive class/function definitions, ten
factories, three DocSpecs and 25 original test functions. The preceding open
items describe earlier checkpoints. Approved Proposals 108 through 115 and the
existing Compiler library and Ecosystem register adapters now supply the
required direct dependencies. Q-0174 and Q-0175 close with the source defects
and execution boundaries below explicitly retained.

`test_addition_foreign_file.py` serializes all ten actual original factory
objects at depths zero and one. Existing Compiler library records preserve
original deterministic bytes, constructor arguments, typed registers, count rows
and decomposition edges. Explicit native factories resolve declarations,
including all three symbolic examples. Symbolic widths retain logical and
model ownership and the original physical-decomposition refusal. The generic
arbitrary source-type graph scope remains in Q-0937.

The three actual concrete Add examples, including 64-bit and unequal 4/16-bit
ports, now use all five existing native example-check modes. Their count mode
compares full emitted opcode multiplicities against independent original
emission. This joins the already accepted four AddK/out-of-place concrete
examples. Existing symbolic source models, dtype/cache identities, input
validation, full matrices, exact adjoints and tensor provider evidence are
reused and rerun. The six complete original unsigned/signed controlled AddK
callback fixtures delegate the existing Simulation constant addition and
controlled-value owners, matching both original root results and decomposed
classical execution. Production numerical and quantum owners remain unchanged.

### SDK arithmetic and sided decomposition contracts

The existing `import_cirq_register_gate` accepts original OutOfPlaceAdder
operations with an explicit typed sided signature. Widths one, two, three and
five, both carry flags and both directions retain reversed SDK Qid placement,
primitive instruction order against the corresponding source execution path,
operand indices, measurement slots, feedback, returned aliases and JSON.
The former universal SDK/component instruction equality claim is superseded by
the source-route correction below: source graph scheduling and the literal SDK
generator differ. Both native paths retain their own exact source stream.
Widths one through three verify every applicable coherent erasure history;
widths one and two additionally verify every basis column with an entangled
spectator. All preserve absolute complex amplitudes and zero consumed bits. Transport produces zero matrix atoms.
Normal append, larger domains and source tensor/isometry checks retain their
accepted owners in the concentrated regression.

Original `ArithmeticGate.registers` produces its three binary radix lists.
Its `apply` returns a, b, c+a+b for either value of is_adjoint. Existing native
integer addition independently matches that unbounded numerical protocol.
The sided clean-output decomposition separately owns creation or consumption
of a correct sum. Original `with_registers` raises NotImplementedError.
Original power-minus-one drops the truncated carry flag, while explicit adjoint
preserves it. These exact source witnesses remain checked. The ordinary static
`import_cirq` adapter retains its unsupported-operation result for the original
root. Register decomposition transport is the executable adapter for this
source contract. Source numerical methods and sided quantum channels retain
their distinct ownership conditions.

### Final validation and resource accounts

Concentrated regression passes 4,236 cases with two existing notebook/environment
warnings and zero skips. Both original and native notebooks run through the
existing complete asset audit. Integration passes 17 cases. All 48 focused
consumer cases also pass with Cirq 1.6.1. The untouched original test file has
267 passes, nine source skips, five source xfails, one notebook deselection and
four source pytest warnings. Its separately accepted notebook evidence is
revalidated in the concentrated asset audit. Strict typing, Black, Ruff, team
scope, architecture, multilevel IR and protected API checks pass for this batch.

All eleven reused production/test owners and the ten files implicated by the
established default-run docstring-policy failure are byte-identical to the
batch baseline. Dependency policy retains 687 existing findings and zero in
changed paths. Whole-repository status retains that documented policy failure. Receipts
and logs use `ftqc-reference-env/g2-addition-foreign-*`.

For a two-bit full sum, actual forward emission is 22 CX, four H, four T,
four TDG and two S, with three caller-visible output bits. Actual inverse
emission is ten CX, two H, two measurements, two conditioned CZ and two resets.
Independent source estimates are two And and ten Clifford for the forward
operation, and twelve Clifford with two measurements for the inverse. Both
source peak estimates are seven qubits. Interchange adds zero instructions,
quantum storage or measurement slots. Existing source count discrepancies stay
recorded independently of these measured streams.

## Complete out-of-place wire graph closure, November 22

The native OutOfPlaceAdder component now records the existing shared
out_of_place_addition_components sequence as typed CX/And wire calls. The
previous hand-built Split/Join/allocation adapter is removed. The already
qualified shared wire reconstruction owns earliest-layer order, lazy word
views and owned sum allocation/consumption in both directions. Numerical gate
recipes, parameters, MSB-first ports, truncation and exact measured erasure
remain with their existing owners. Callers still use circuit.append(op).

The source stream test now requires the complete ordered native trace to equal
the frozen source trace for all twenty width/output/direction cases. The former
canonical edge comparison that ignored instance IDs is removed. Widths 1, 2, 3,
5 and 8 cover both output-width options and both directions. Existing complete
small complex isometries, entangled spectators, every applicable erasure
history, valid sum cleanup, ownership errors and independent source resource
formulas retain their existing assertions.

This closes six recursive comparison dependency mismatches, comprising the
controlled comparator's forward/inverse adders and four half-comparison adders.
Recursing through all twelve concrete comparison catalog roots now finds 62
exact source-success graphs and 344 atomic/refusal contexts, with no remaining
ordered-call graph mismatch. These counts retain owning root contexts. They do
not imply 62 distinct upstream classes or qualification of recursive glyphs.
Original source files and source atomic/decomposition failures are unchanged.

The concentrated execution, comparison, root drawing and live-width regression
passes 788 cases. Strict typing of three changed Python files, Black, Ruff,
Compiler scope, architecture, IR, public API and whitespace checks pass. All
twelve concrete comparison factory primitive histograms equal the published
baseline exactly. Actual primitives and independent analytical estimates remain
separate. Recursive glyph publication remains the next presentation boundary.

The presentation adapter now directly reuses the executable OutOfPlaceAdder
owner. Its previous extra graph-copy step is removed. The frozen Kickmix
addition/comparison file checks pass all 26 cases with zero skips.

## Complete recursive addition presentation, November 22

The addition.py file's seven concrete catalog roots now qualify every recursive
source-success child drawing. Its complete walk contains 13 exact ordered graphs
and 61 atomic/refusal contexts. These counts preserve owning root contexts.
The symbolic refusals retain the earlier source evidence. Original source code
and its drawing failures remain unchanged.

The Arithmetic exporter supplies a domain-owned child presentation callback to
the existing dependency publisher. Its addition provider identifies Add,
Always(Add), XorK and both OutOfPlaceAdder directions only after full reachable
native definition matching. Declared constant metadata supplies XorK's encoded
pattern. The provider reuses the existing executable constructors and Drawer
symbols. It introduces no alternative executable or resource implementation.
Stored dependency identities, graph edges and publication filenames are retained.
Other domains continue to use their existing provider.

The fresh source comparison exposed a real distinction between the AddK root's
Always label and the Add child inside Always. The latter requires the existing
addition wire glyph, including its blank vertical label and a+b output port.
That callback is now reused for the matching Always body. XorK's individual X
children reuse the Pauli X glyph. All seven roots' recursive source-success
children are checked against fresh source diagrams. The frozen source still
raises IndexError for the width-100 Add child. The native drawing remains
available and that original failure is retained explicitly. Twelve small
OutOfPlaceAdder cases additionally cover both directions and output widths.
A mutated T phase in either reachable Add or Always body fails qualification.

The Arithmetic dependency inventory contains 2832 contextual native functions.
Its qualified labels increase from 16 to 39, making 61 in the combined catalog.
The 23 additional matching definitions include addition dependencies reused by
other Arithmetic roots. Literal constructor matching is recorded separately
from the fresh source drawing checks of the complete addition file. The combined
catalog retains 3569 JSON files, 234 documentation files and 3568 diagrams.
Physical gate streams, full complex branch and workspace evidence, and separate
analytical models reuse the preceding 788-case execution and 26-case locked
Kickmix acceptance because this batch changes presentation only. Other recursive
Arithmetic glyphs and EqualsAConstant's nominal Join view remain open.

The concentrated catalog regression first failed with ENOSPC. Its original log
is retained as g2-addition-child-catalog-disk-failure-20261122.log. Only the
current failed export outputs and an older confirmed reproducible pytest export
directory were removed, releasing 188344893 logical file bytes. No source,
reference checkout or user archive was deleted. The source OutOfPlaceAdder
operation label is retained in both directions because its adjoint is a
constructor flag rather than the generic Adjoint wrapper. The six matching
comparison child drawings and their source labels also pass fresh comparison.

After cleanup the concentrated catalog run passes 586 checks, with three
pre-existing deselections and fourteen existing notebook escape warnings. The
final 32-case focused run includes all twelve comparison roots and all qualified
recursive children encountered there. Strict typing of five changed Python
files, Black, Ruff, integration scope, architecture, IR, public API and whitespace
checks pass. The earlier 788 execution and 26 Kickmix checks remain the physical
acceptance evidence. Primitive counts and analytical resource estimates remain
reported by their original owners.

## Exact SDK and graph source-route correction (2026-11-27)

The complete shared-fixture regression exposed six failing adjoint cases in the
old SDK/component equality assertion: widths 2, 3 and 5 with either output flag.
The unchanged source itself has distinct linear schedules. For the two-bit
truncated adjoint on wires 0 through 5, the literal SDK generator's second
operation is CX(0,4), while Cirq's graph schedule places CX(3,5) second. A new
source-only witness calls the original generator and original Cirq scheduler;
it uses no FlagQuantum transformation or reordered comparison result.

Three owners are checked independently over widths 1, 2, 3, 5 and 8, both output
flags and both directions in the diagnostic evidence:

| Existing native owner | Exact unchanged source counterpart |
| --- | --- |
| `out_of_place_add_ir` | Literal `decompose_from_registers` sequence |
| `out_of_place_addition_program` | CompositeBloq graph/export sequence |
| Pure SDK Program import | Source graph schedule |
| Measured SDK Program import | Literal source feedback sequence, kept as one body |

The original comparison incorrectly required the last two transport modes to
equal the component graph's instruction tuple in every direction. The corrected
complete test retains exact instruction equality for each matching source path,
including all operands and feedback slots. It also checks identical primitive
counts, original source model totals, output ownership, extent, JSON and absence
of matrix fallback. No instruction sorting, canonicalization, tolerance widening,
reference patch or expected-failure waiver is used.

Both native paths now run each small measurement history on the same full
complex coherent input with a spectator through three bits. Every basis column
and history is also checked through two bits. Both must match the independent
addition/isometry oracle with its unnormalized branch amplitude and clean
consumed sum; agreement only up to global phase is insufficient. Larger widths
have exact stream/model checks, not dense-state execution claims. Primitive
T/TDG, measurement and reset counts remain separate from the unchanged model's
And/Clifford categories, total T estimate and logical-qubit estimate.

This corrects a test's execution-path assumption after the existing earliest-layer
wire-graph integration. No arithmetic, scheduler, SDK import implementation or
upstream source is changed by the correction. The source truncation-loss power
case and other retained source defects stay open under their existing scopes.
