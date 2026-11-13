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
