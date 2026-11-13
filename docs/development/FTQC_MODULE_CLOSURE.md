# Fixed integration module closure inventory

The G2 scope remains the original **1,280 upstream files**: 945 Qualtran and
335 Kickmix files. It does not grow with test scenarios. See
[the frozen inventory](ftqc_source_inventory.csv), SHA256
`62970fe28e322a8d8f0f1921cf152d36a89eb96431cb474aa6116dce63f2c98e`. Original IDs, pinned revisions and Git blob identities are
preserved. Qualtran is v0.7.0; Kickmix is the separately pinned subtree revision.
The source-audit and definition-navigation columns are the initial audit snapshot,
not live acceptance decisions or an assertion that unlisted behavior is absent.
Unavailable reference assets remain open; differing blobs cannot certify an
unchanged upstream comparison. G3 integration acceptance remains separate.

## Module closure criteria

1. Audit the whole source file: functions, classes, methods, properties, constants,
   exports, dependencies, examples, tests and repository assets. AST names are
   navigation only. Every capability needs a native target or a justified asset
   disposition; a child checkbox never automatically closes a parent.
2. Establish native ownership, interfaces, bit order and work ownership before
   implementation. Reuse existing components and preserve source copyright.
   Do not mirror the upstream package or create another IR.
3. Compare unchanged upstream behavior, including failure paths. For emitted
   circuits, compare ordered operations, operands, feedback, exact complex
   amplitudes, small exhaustive measurement branches and clean workspace.
   Numerical-only modules instead verify values, precision, warnings and state.
4. Report measured primitives, source cost models and runtime statistics
   separately. Unimplemented variants, source defects or protected-contract gaps
   keep the file open. Do not silently repair the reference.
5. Commit complete modules independently; run concentrated regression once at
   batch end. Publish locally verified work to #1084 without waiting for CI.
   Do not merge or start G3. Retain baseline failures explicitly.

## Frozen execution order

| Order | Original parent groups | Completion unit |
| --- | --- | --- |
| 1 | Q-0757 / Q-0758 | Both matrix functions and the original test file |
| 2 | Remaining Q07 linalg/simulation | Whole numerical modules, filling actual gaps |
| 3 | Q01 / Q04 | Shared register/graph contracts and resource modules |
| 4 | Q02 / K01-K03 | Arithmetic, field, table, swap and synthesis modules |
| 5 | Q03 / Q05 | Algorithms, correction and cost models using closed dependencies |
| 6 | Q06 / Q07 drawing / K04-K06 | Formats, simulation, interoperability and tools |
| 7 | Q08 / K07 | Packaging, build, docs, licensing and repository asset dispositions |

Pause further GQSP scenario expansion while closing this backlog. Newly found
behavior belongs to an existing parent ID; any scope change needs an explicit
record. The Chinese workspace ledger remains the live status source and its
phase totals are generated from parent decisions, not test count.

## Closed source modules

[Matrix numerics](MATRIX_NUMERICS.md) closes Q-0757 and Q-0758. The parent ledger
now also closes [polynomial evaluation and matrix assertions](POLYNOMIAL_ASSERTION_CLOSURE.md)
(Q-0762, Q-0763, Q-0767), and the [Jacobi-Anger original test file](JACOBI_ANGER_SOURCE_AUDIT.md)
(Q-0765), and the [permutation original test file](PERMUTATION_SOURCE_AUDIT.md)
(Q-0760). The [alias preprocessing audit](ALIAS_DISCRETIZATION_SOURCE_AUDIT.md)
closes Q-0754, Q-0755, Q-0756 and Q-0761, including two explicitly inspected
namespace-only assets. The [register tensor original test file](REGISTER_TENSOR_SOURCE_AUDIT.md)
(Q-0894) and the [dense original test numerical replacement](DENSE_AXIS_SOURCE_AUDIT.md)
(Q-0888), plus the [tensor protocol original tests and namespace assets](TENSOR_PROTOCOL_SOURCE_AUDIT.md)
(Q-0880, Q-0886, Q-0890, Q-0896), followed by the
[complete classical/tensor notebook and five-function test audit](CLASSICAL_TENSOR_NOTEBOOK.md)
(Q-0897, Q-0899), bring the total to **19 closed files and 1,261 open files**. These file
counts are not a functional completion percentage or an effort estimate. Some open modules
already have accepted partial capabilities; many require full audit.


## Audited source with an inherited open defect

The [QSP inspection audit](QSP_INSPECTION_CLOSURE.md) covers all five functions
in Q-0766, including default FFT, NumPy scalar precision and failure-domain
inputs. Its finite-grid assertion still misses a source counterexample with
between-grid magnitude 1.06. Per the frozen criteria, the parent stays open
rather than treating interface completion as repair of that upstream defect.
No additional scenario or parent ID was added to the completion denominator.


The [Jacobi-Anger source audit](JACOBI_ANGER_SOURCE_AUDIT.md) fills its default
Bessel interface and observes eight exceptional search inputs in bounded
processes. Q-0765 closes after all 24 original tests also pass with native
bindings. The source implementation parent Q-0764 stays open for the inherited
search/heuristic limitation under the same frozen closure rules.

The [permutation source audit](PERMUTATION_SOURCE_AUDIT.md) covers dense graph
discovery and ordered partial traversal, reusing the existing append emitter.
Q-0760 closes after both original tests run unchanged and with native bindings.
Q-0759 remains open for inherited partial-map prefix loss and its invalid
keyword example; source behavior is preserved rather than silently repaired.

The [alias preprocessing audit](ALIAS_DISCRETIZATION_SOURCE_AUDIT.md) assigns all
six source functions and width overloads to one Algorithms owner. Existing
validated concrete/symbolic factories replace duplicate algorithms with shared
calls. All original tests run unchanged and with native bindings. Namespace
assets close by explicit AST/blob disposition, without recreating upstream
package names or claiming new algorithm functionality.

The [register tensor data audit](REGISTER_TENSOR_SOURCE_AUDIT.md) maps all six
functions to a shared numerical owner, replacing duplicate projection and control
slice loops. Q-0894 closes after all original tests pass unchanged and with native
bindings. Q-0893 stays open for inherited paired-directional-name reconstruction
and scalar object-array failures. Native validated paired-name support remains
explicitly separate from raw source behavior.

The [dense axis audit](DENSE_AXIS_SOURCE_AUDIT.md) closes the complete original
test-file mapping through native numerical-boundary replacement. The unchanged
source graph and flattening remain in the test adapter; native graph consumers
are checked separately. Q-0887 stays open for width-triggered simplification,
and Q-0889 remains open for its independent flattening gaps. All phase and
section totals are synchronized without adding scenario IDs.

The [tensor protocol audit](TENSOR_PROTOCOL_SOURCE_AUDIT.md) executes the complete
flattening/classical-action original tests with native boundary replacements.
Two namespace files close by complete AST and export responsibility disposition,
not by treating their referenced implementations as complete. Q-0889 and Q-0895
remain open for their explicit source/native boundaries. Five numerical
consumers now import the authoritative shared axis owner directly; non-import
production ASTs are unchanged. No new child ID or completion denominator is
introduced.

The [classical graph execution slice](COMPONENT_CLASSICAL_STATE.md) connects the
existing numerical binding, action, distribution and phase owners to Core SSA
call stepping. Exhaustive Add assignment traces and X-measurement/coherent phase
consumers agree with unchanged source. Q-0882 and Q-0883 remain open for
finite-field/dtype traversal, source-wide utilities, native admission boundaries
and original test replacement. The 28 original tests are source-only regression.
No new child ID is added; the totals remain 17 closed and 1,263 open files.

The [classical truth table slice](CLASSICAL_TRUTH_TABLE.md) maps both scalar
truth-table enumeration and formatting onto existing Core metadata, numerical
domains and bound evaluators. The complete Add table reconstructs the source
matrix and matches every measured forward branch with clean work. Q-0882 and
Q-0883 remain open for source-wide graph and original test replacement gaps.
The frozen 1,280 parents and 1,172 checked child records remain unchanged.

The [field graph value slice](CLASSICAL_FIELD_REGISTERS.md) connects explicit
Simulation field encoders to already admitted binary-field Core ports. Integer
and field consumers replace separate traversal with shared binding/collection;
field storage and validators stay with the existing numerical codec. Complete
field-add tables/matrices, original gate operands and complex append roundtrips
pass. Q-0882 and Q-0883 remain open for source-wide resolution and original test
replacement. No additional parent or child acceptance count is inferred.

The [binary metadata resolution slice](BINARY_FIELD_ENCODING.md) restores existing
numerical field encoders from Core-admitted descriptors and removes the manual
mapping requirement for those graphs. It retains polynomial/coefficient-field
identity, exact source field classes and explicit override precedence. It is a
native representation adapter, not a source-wide dtype or admission expansion.
Q-0882/Q-0883 and fixed parent/child counts remain open and unchanged.

The [bit graph validation fix](COMPONENT_BIT_VALUES.md) removes integer validation
from bit ports and reuses the existing equality-based bit owner. Three concrete
counterexamples now match unchanged source acceptance/rejection; floating and
complex fixed outcomes retain exact phase and physical branch amplitudes.
Q-0882 and Q-0883 remain open; frozen parent and child counts do not increase.

The [recursive failure context](COMPONENT_CLASSICAL_RECURSION.md) preserves
source NotImplementedError cause chains through explicit native recursive
callbacks. Direct actions, other error categories and output commits retain
their existing failure boundaries. Real depth 2/3/5 graphs match source chain
contents without committing failed parent outputs. Foreign decomposition error
translation and source-wide closure remain open; acceptance counts do not rise.

The [opaque graph value fix](COMPONENT_OPAQUE_VALUES.md) removes integer-domain
validation from QAny-equivalent ports while retaining shared shape traversal,
sampling and source NumPy storage conversion. Scalar identity, array conversion
failures/warnings, RNG state and complex physical identity append consumers are
compared to unchanged source. Source fixed-input handler failure is retained as
an explicit difference. Q-0882/Q-0883 and fixed acceptance counts remain open.

The [recursive phased evaluator](COMPONENT_PHASED_EVALUATION.md) removes
caller-written state reconstruction for stored typed bodies. Compiler-supplied
call orders feed the existing state; nested states inherit the same random
handler and return phase to their parent. Original composed phasing, nested
measurement branches and complete Add input tables agree with source. Pure
recursive random-policy and foreign decomposition boundaries remain open;
no whole-file completion or larger denominator is inferred.

The [recursive instance-ID binding](COMPONENT_PHASED_INSTANCE_IDS.md) connects
Compiler's existing persistent expansion labels to the state owner's fixed
measurement lookup at each function. Root and nested graphs agree with original
source instance IDs, including reversed storage versus visitation and local-ID
reuse. Full complex two-measurement branches retain their projection factors
and clean spectator. Raw source CompositeBloq pure dispatch remains a separate
negative; parent closure counts remain unchanged.

The [pure recursive evaluator](COMPONENT_PURE_EVALUATION.md) supplies the missing
source default-RNG policy for stored bodies. Root policies remain local while
nested calls reuse the process default, independently of phased recursion.
Full complex Add branches, exact primitive streams and separate QEC model
accounts agree with the frozen source. Source-wide dtype and foreign exception
boundaries remain open; parent and child counts do not increase.

The [classical register codec composition](CLASSICAL_REGISTER_CODEC.md) connects
17 quantum/classical descriptor cases to existing numerical owners without
expanding Core execution admission. Default dense tables replace handwritten
codec dispatch, preserve source Montgomery/negative-zero/empty-array behavior
and match full complex Add branches. Q-0895 remains open for wider domains and
its original classical-axis/phase-free boundaries; inventory counts do not rise.

The [binary field codec connection](CLASSICAL_FIELD_CODEC.md) extends the shared
metadata adapter through existing field resolution. Typed field codecs, input
effects and exact Add matrices/primitive streams match the frozen source.
Default dense identity output still loses its field class at the original
NumPy conversion and fails; multidimensional array limitations are retained.
Q-0895/source-wide closure remains open and the inventory counts do not change.

[Shared field metadata reconstruction](FINITE_FIELD_METADATA.md) replaces the
binary-only numerical inverse with one quantum/classical field owner. The
executable binary entry still calls Core admission first. Odd-field/classical
codecs now match source domains, full multiplication tables, types and failures;
this adds no executable gate family. Original dense and source-wide boundaries
remain open and fixed inventory counts do not increase.

Concrete register domains and the default truth-table connection reuse existing numerical owners; see [Classical register domains](CLASSICAL_REGISTER_DOMAINS.md). No executable admission is expanded.

[Typed classical values](CLASSICAL_TYPED_VALUES.md) connect concrete declarations to existing validation and collection owners, replacing state-local dispatch without widening executable admission.

[Symbolic numerical metadata](CLASSICAL_SYMBOLIC_METADATA.md) reaches existing binding, collection and domain owners while binary codecs retain concrete admission.

[Finite-field polynomial codecs](FINITE_FIELD_POLYNOMIAL_ENCODING.md) reuse existing field and array numerical owners; polynomial layouts and gate emission remain in Compiler, and Core admission is unchanged.

[Polynomial numerical metadata](FINITE_FIELD_POLYNOMIAL_METADATA.md) connects concrete declarations to shared domain, codec, binding and collection owners; Core executable admission remains unchanged.

[Integer basis tensor suppliers](REGISTER_BASIS_VECTORS.md) provide arbitrary unsigned kets/effects on existing labels and replace the zero supplier implementation; Compiler retains physical preparation and graph construction.

[Classical vector graph construction](COMPONENT_CLASSICAL_VECTORS.md) composes existing typed ownership, integer loading and numerical vector suppliers. Its explicit evaluator boundary keeps Compiler independent of Simulation; wider source dtype and failure interfaces remain open.

The original X, CNOT, Toffoli and five-bit Add classical/tensor examples now run through native executable graph replacements; complete matrices, complex measurement branches, work cleanup and separate resource accounts pass. Q-0899 remains open for its notebook test; Q-0898 retains source-wide dtype and diagnostic gaps. See [Classical vector graph construction](COMPONENT_CLASSICAL_VECTORS.md).

[Factorized addition programs](ADDITION_COMPONENTS.md) share one literal logical
sequence with the existing physical forward/exact-adjoint emitters. The original
32-bit notebook scalar contracts through native numerical suppliers with 32-element
peak intermediates. Native notebook drawing/provider composition remains open;
original notebook execution is recorded separately from native replacement.

## Core graph drawing slice

[Component register scores](COMPONENT_REGISTER_SCORE.md) now connect existing
Core typed calls to the existing drawer. Original notebook closed diagrams and
expanded Add through 32 bits have matching geometry and explicit glyph suppliers;
native identifiers and open-boundary text differ. Q-0747/Q-0749/Q-0897 remain
partial: source-wide drawing and automatic numerical provider composition are
not closed by these cases. The fixed 17-file closure total is unchanged.

## Explicit Add numerical composition

[Add tensor providers](ADDITION_TENSOR_PROVIDERS.md) replace test-only numerical
dispatch with an explicit example binding existing suppliers. Split/join and
through-identity nodes share Simulation bit-wiring logic. Source-wide provider
coverage and notebook parent closure remain open; no additional parent file is
closed by this cleanup.

## Classical-vector test dependency cleanup

The [shared scenario checks](COMPONENT_CLASSICAL_VECTORS.md#shared-scenario-checks)
move reusable observation/contraction helpers out of a collected test module.
All 85 affected collection IDs and original assertions survive. Product code
and source-module acceptance decisions are unchanged; this cleanup adds no new
capability checkbox or closed parent file.

The [shared construction fixtures](COMPONENT_CLASSICAL_VECTORS.md#shared-construction-fixtures)
continue that cleanup through finite call graphs, private workspace and locked
classical-vector examples. Six moved functions and 241 existing scenarios retain
their implementation/collection contracts. Wider strict typing has 55 confirmed
unchanged baseline diagnostics; this refactor closes no additional source file.

The [binding-test type closure](COMPONENT_CLASSICAL_VECTORS.md#binding-test-type-closure)
resolves the 55 diagnostics recorded in the previous fixture cleanup. The same
14-file surface passes strict typing and the same 241 existing cases pass;
original assertions and decorators remain. This is a test-only maintenance
result, with no additional source-file closure or resource-model claim. The
previous default-suite timeout remains unresolved.

The [canonical basis recipe query](ADDITION_TENSOR_PROVIDERS.md#read-basis-recipes-from-canonical-definitions)
removes the external value side table from the native 5-/32-bit closed Add
notebook checks. Compiler recognizes only its exact integer-vector definitions;
Simulation continues to supply their numerical tensors. This extends the
existing Q-0897 composition acceptance without closing the source-wide notebook
or provider parent files. Source data, physical decompositions, measured branch
semantics and resource models are unchanged.

The [unitary register tensor bridge](REGISTER_UNITARY_TENSORS.md) adds the
PyTorch direction-projection path for the original gate-to-tensor chain, reusing
one zero-slice constructor with the raw NumPy owner. Complex phase and gradients
survive explicit provider composition and physical append execution. It extends
Q-0893's projection evidence without closing that source parent or the foreign
gate/provider/notebook parents. Actual primitive gates and the locked QEC model's
rotation estimates remain separate.


The [complete native notebook workflow](CLASSICAL_TENSOR_NOTEBOOK.md) closes the
concrete Q-0897 notebook asset and Q-0899 original test file after full audit.
The original 13 code cells execute through native boundary adapters; a native
13-cell notebook independently renders five diagrams and contracts the original
Add(32) input. Nine Markdown cells, source attribution, kernel metadata and the
absence of stored output assets are accounted for. All five original tests run
unchanged, with corrected same-process import isolation. This raises the live
parent closure total to 19 and leaves 1,261 open; the 1,181 checked capability
records do not change. General provider/foreign-interface implementations,
planner performance and the default-suite timeout remain open. Earlier counts
in batch narratives above are historical snapshots, not the live total.

The [factored controlled-NOT leaves](FACTORIZED_CONTROLLED_NOT.md) add the
source COPY/XOR recipe and expose the existing pure graph's internal-label
allocator to explicitly declared suppliers. The recovered Q-0885 notebook blob
is now available for further audit; its complete workflow remains open. This
accepts Q-0231.04 and extends existing Q-0891 wiring evidence, bringing checked
capability records to 1,182 (1,181 current plus one historical). Complete parent
closure remains 19 accepted and 1,261 open. Node counts, physical CX counts and
source QEC models are measured separately; no performance parity is inferred.

The [native tensor incidence view](TENSOR_NETWORK_INSPECTION.md) accepts the
bounded Q-0885.01 inspection dependency, preserving every source CNOT tensor-axis
connection and friendly/structured outer name. Drawer consumes existing nodes and
side tables and adds DOT/Matplotlib presentations without numerical execution.
Full 18-cell notebook acceptance remains open. Checked capabilities rise to
1,183 (1,182 current plus one historical); parents remain 19 closed and 1,261
open. No resource-estimate, physical-gate or performance claim changes here.

The [source-ordered conjunction provider](CONJUNCTION_TENSORS.md) accepts
Q-0545.05 and extends the existing Q-0885 notebook dependency evidence with real
And/MultiAnd composition. Exact five-axis leaves, all two-to-four-control
activation patterns, complete complex isometries and every small erasure history
pass. Existing compiler emitters and physical/model accounting remain separate.
Capabilities rise to 1,184 (1,183 current plus one historical); full parents
remain 19 closed and 1,261 open. Bell/custom-provider composition and the complete
18-cell notebook remain open; no G3 or performance acceptance is implied.

The [Bell/custom-tensor workflow](BELL_TENSOR_WORKFLOW.md) extends Q-0885's
existing dependency evidence through direct PlusState/ZeroState/CNOT preparation,
H/CNOT unitary, explicit zero initialization and unchanged original MyCNOT/Bell
notebook definitions. Dense/factored suppliers share the consumer and physical
program; missing unflattened suppliers fail. Counts remain 1,184 capabilities
and 19 closed / 1,261 open parents. Complete notebook, custom MultiAnd flattening
and contraction-info acceptance remain open; this composition adds no duplicate
capability record.

The [instance/contraction-info workflow](CONJUNCTION_CONTRACTION.md) extends
Q-0885's existing dependency evidence with original MultiAnd occurrence IDs,
logical atomic-leaf stopping and explicit native profile inspection. Imported
source paths reuse existing pair-step conversion/profiling/execution; matching
peaks and distinct FLOP/native-cost conventions are verified. There is no new
capability ID or parent closure: 1,184 records, 19 closed and 1,261 open remain.
Whole 18-cell notebook and original index/layout asset review remain open.

The [complete tensor notebook audit](TENSOR_SIMULATION_NOTEBOOK.md) now closes
Q-0885 as a concrete repository asset: all 18 original code cells and all 18
native cells execute, all 18 Markdown cells and kernel/empty-output/attachment
metadata are disposed, and five native figures are rendered and visually
reviewed. Original numerical code remains unchanged; only the original optional
drawing boundary is intercepted because Graphviz is unavailable. Native integer
edge inspection and Matplotlib layouts are explicit adaptations, not foreign
object or pixel equivalence. Nine complete complex arrays and exact leaves,
selection, diagnostics and profile accounting pass. Counts are now **20 closed,
1,260 open parents**, **390 active**, with **1,184 capability records unchanged**.
The related 1,945 tests and 84 CPU/TN tests pass; strict checks cover the new test
file. Existing physical/measurement/workspace evidence is preserved by those
regressions. Other source-wide tensor/optimizer interfaces and the prior default
suite timeout remain open. G3 and integration-PR merge are not authorized here.

The [source-ordered rank pass](TENSOR_RANK_SIMPLIFICATION.md) extends Q-0887.03
with the R dependency of Qualtran's width-triggered numerical simplification.
The trigger is in `_dense.py`, not `_quimb.py`; Quimb is unpinned by Qualtran,
so tests explicitly fix the reference dependency to 1.15.0 and verify its source
hashes. Queue choices, operands, retained hyperedges, complete complex results,
12/13-bit threshold behavior, meta structure and gradients pass. Scalar
absorption into one native leaf is explicitly distinguished from source
fractional-power spreading. The 71 new / 2,016 related and 84 CPU/TN tests pass;
module/example coverage is 100%, strict checks cover three Python files.
Squeeze/A/D/C and full default-policy integration remain open. Counts stay
20 closed / 1,260 open parents, 390 active, and 1,184 capabilities; this dependency
adds no duplicate child ID or parent closure. No G3 or merge is performed.

The [squeeze/C dependency](TENSOR_COLUMN_REDUCTION.md) extends Q-0887.03 without
a new ID or parent closure. Shared output validation is reused by R; squeeze
protects output extents, and C matches original cut order and every resulting
leaf on admitted finite NumPy-reference data. The existing slicer now removes
all repeated-label axes. Fixed-support/Bell phase gradients pass; explicit
counterexamples retain the source's tolerance-amplified error and changed
ambient zero-entry Jacobian. Source NaN/backend and empty-shape differences are
recorded separately rather than silently repaired. There are 130 new, 2,146
related and 84 CPU/TN passing tests; new numerical owners/example cover 96
statements and 28 branches at 100%, and strict checks cover seven Python files.
A/D and complete default policy remain open. Counts remain 20 closed / 1,260
open parents, 390 active and 1,184 capability records. No G3 or merge is performed.

The [A/D dependency](TENSOR_DIAGONAL_REDUCTION.md) extends Q-0887.03 with
source-ordered antidiagonal gauge and diagonal reduction, sharing finite support
with C and preserving captured register output groups through hyperedge creation.
Repeated-index upstream gauge defects, tolerance errors and ambient-gradient
boundaries remain explicit. Fixed-point composition and full width triggering
remain open; no duplicate capability record or parent closure is added.

A/D acceptance: 146 new / 2,292 related / 84 CPU-TN tests pass; strict checks
cover eight changed Python files, and new numerical owners plus example cover
122 statements and 40 branches at 100%. Implementation `051cde78`. Counts stay
20 closed / 1,260 open / 390 active parents and 1,184 capability records.

The [default ADCR fixed point](TENSOR_FIXED_POINT.md) extends Q-0887.03 using
existing pass owners with run-local negative caches, stable occurrence ownership
and ordered adjacency. Source/native every-operation and per-round count traces
are compared, including a three-round sparse graph; source zero-scalar NaNs
remain explicit. Complete width planning/automatic policy and the parent remain
open. Counts and capability IDs are unchanged.

ADCR acceptance: 60 added / 2,352 related / 84 CPU-TN tests pass; strict7,
new loop/example57 statements12 branches100%. Implementation `0f1dbb2b`.
The source width planner and complete automatic entry remain open.

The [width planning dependency](TENSOR_WIDTH_POLICY.md) extends Q-0887.03 by
reusing existing optional Cotengra at the locked source-default greedy boundary.
Explicit simplification_atol selects ADCR above reported width12 with groups
captured first, through the same register/component consumers. Source singleton
raw-size defect and all inherited gradient/source differences remain explicit.
Default consumers remain unsimplified; full default-entry policy and parent
closure remain open, with unchanged capability and parent counts.

Width acceptance: 43 new / 2,395 related / 84 CPU-TN pass; strict7 and
width/full-loop/example86 statements24 branches100%. Implementation `73816a0d`.
No parent closure: 20 closed / 1,260 open / 390 active; 1,184 capability records.

Dense entry acceptance: 30 new / 2,425 related / 84 CPU-TN pass; strict4
and format/scope/architecture/IR/API/example. Implementation `84e51b4c`. All four
source functions have explicit disposition, but source defects/default-gradient
differences retain Q-0887 open. Counts remain20 closed/1260 open/390 active,
1184 capability records. Next: Q-0889 whole-file flatten disposition.

Flatten disposition: 23 new / 2,502 related / 84 CPU-TN pass; strict1
and format/scope/architecture/IR/API/example. Acceptance `e6ad2789`. All two
source functions and delegated error semantics have explicit disposition;
foreign reflection/lazy callback boundaries retain Q-0889 open. No new
production layer or capability ID;20 closed/1260 open/390 active,1184 records.

Zero/discard: RIGHT classical output guard corrected, implementation `32f4e4bd`.
8 new / 2,632 related / 84 CPU-TN /2 original source tests pass, strict2
and format/scope/architecture/IR/API/example. Q-0891 retains source empty/zero
width/direct-discard differences and physical pool-retirement refusal; Q-0891.04
stays unchecked. No new ID;20 closed/1260 open/390 active,1184 records.

Classical tensor disposition: acceptance `cd1648e1`,30 new/2692 related/
84 CPU-TN/6 unchanged source tests pass; strict1 and quality checks; owner
52 statements10 branches100%. All4 functions/nested iterator have targets,
but classical-rank/empty/phase limitations keep Q-0895 open. No new ID;
20 closed/1260 open/390 active,1184 records. Next Q-0882; reuse Q-0893 audit.

Classical simulator: complete definition ownership is recorded in
[the source disposition](CLASSICAL_SIM_SOURCE_DISPOSITION.md). Implementation
`c629405e` corrects current-output measurement phase lookup after output commit.
13 new /12,943 related /84 CPU-TN /28 unchanged original tests pass; strict2,
format/scope/architecture/IR/API and two examples pass. State coverage97%;
original notebook and native replacement matrix remain distinct acceptance work.
Q-0882/Q-0883 remain open; no new ID or physical primitive.
20 closed/1260 open/390 active,1184 records remain unchanged.

Classical simulator test matrix: all19 original test functions are mapped in
[the complete matrix](CLASSICAL_SIM_TEST_MATRIX.md);22 original parameter cases
execute unchanged assertions against native entries, six functions retain
explicit parallel representation checks. Test commit `f6076c45`;34 new/1591
related/84 CPU-TN/28 original source-only checks pass, strict1 and quality gates.
Q-0881 reference notebook is restored byte-for-byte; its12-code/13-Markdown
asset migration remains open. Q-0882/Q-0883 are not closed by test counts.
No new ID:20 closed/1260 open/390 active,1184 records remain unchanged.

The [complete classical simulation notebook](CLASSICAL_SIMULATION_NOTEBOOK.md)
closes Q-0881: all 25 cells (12 code/13 Markdown), metadata and assets have
explicit native disposition. Original 12 code cells and native 12 code cells
execute independently; per-call Add outputs and the physical two-bit consumer
match, including both unnormalized measurement branches and clean workspace.
Implementation `ce8bc24a`; 6 new/1622 related/84 CPU-TN/28 unchanged source tests
pass, strict2 and quality gates. Native value-table plotting and the bounded
exact fixed-point sample are explicit adaptations, not general QFxp support.
Q-0882/Q-0883 remain open. Current total: 21 closed/1259 open/390 active,
1184 capability records (1183 current plus 1 historical). Next Q-0884.

The [complete superoperator notebook](SUPEROPERATOR_NOTEBOOK.md) closes Q-0884:
all 28 cells (13 code/15 Markdown), metadata and six figures are disposed.
Unchanged original code and independent native code agree on complete complex
states/channels; every coin/dilation/discard tensor and axis connection matches.
The physical H-CZ-H isometry, both measured branches, clean consumed carriers,
spectator and ordinary append pass. Implementation `37922753`; 6 new/277
related/84 CPU-TN checks, strict2 and quality gates pass; six frozen blobs match.
Helper coverage reaches98% with its standalone example. Diagram layout and
physical materialization are explicit adaptations; analytical QEC models and
emitted primitives remain separate. No protocol implementation parent closes.
Current total:22 closed/1258 open/390 active,1184 capability records unchanged.
Next Q-0879 complete MBUC notebook audit, after restoring its frozen reference.

The [measurement-uncomputation notebook](MEASUREMENT_UNCOMPUTATION_NOTEBOOK.md)
has complete23-cell disposition, ten unchanged original and ten native code-cell
executions, and source/native every-call values and phase checks. Implementation
`98117c39` reuses existing exact AND adjoint and phased SSA owners. Corrected and
naive emitted streams, both unnormalized complex branches, consumed-target zero,
spectator entanglement and append pass. Seven new/1629 related/84 CPU-TN checks,
strict2 and quality gates pass; six frozen blobs match; helper97% with example.
The source's stale-phase assertions, global debug-state lookup and unavailable
C[CZ] resource decomposition remain explicit, so Q-0879 stays open. No new ID;
22 closed/1258 open/391 active,1184 capability records remain. Next Q-0898 whole
classical/tensor cross-check source audit; do not repeat its closed test assets.


### Classical-vector source failure audit (2026-11-12)

All three Q-0898 functions and dependencies have native ownership dispositions in
CLASSICAL_VECTOR_SOURCE_AUDIT.md. Commit 405f8802 corrects scalar array-effect
refusal before definition or word consumption, including empty shapes. Twenty-five
new cases, 2365 concentrated and 84 CPU-TN checks pass; strict2 and all quality
gates pass, helper coverage100%, six frozen blobs match. The measured Add
consumer covers64 inputs/16 histories, complex amplitudes, work zero and append.
No new primitive/workspace; emitted wrapper X0–11 and source model Clifford6 are
separate. Source empty automatic evaluation and wider constructor acceptance,
foreign protocols and executable admission keep Q-0898 open. No new capability
ID:22 closed/1258 open/391 active,1184 records. Next Q-0736 classical assignment
graph audit, retaining existing Simulation/Drawer ownership.


### Classical producer-value graph (2026-11-12)

Q-0736's three methods map to explicit Simulation assignments and existing Drawer
geometry. Implementation312df299 adds producer SSA labels without numerical
execution. Two polarities/all16 inputs/13 edges/styles, THRU0/1/0, empty/shaped,
errors and real Graphviz rendering pass. Eleven new/2919 related/84 CPU-TN tests,
strict3 and all quality gates pass; both owners100%, six frozen blobs match.
Q-0737's sole original test has unchanged assertions under native replacement and
source-only execution; that test-file parent is closed. Q-0736's foreign/base
admission remains open. Actual H6/T6/Tdg6/CX18/S3 and modelAND3 are separate;
drawing adds no gates or work and leaves the bound physical program identical.
23 closed/1257 open/392 active,1186 capability records. Next Q-0741 Graphviz
notebook: restore the frozen source before whole-cell audit, retaining existing
Simulation/Drawer ownership. No G3/merge/default-timeout retry.


### Graphviz notebook signature and executable paths (2026-11-12)

Q-0741's21 cells/13code are dispositioned; unchanged source and native notebook
execute independently. Commit1fbfdecb adds standalone signature drawing without
executable placeholders; all original100/200-bit and opposite-side shaped ports
remain. Split/3X/Join connections and complete complex matrix/spectator-zero/
append pass. Twelve new/2931 related/84 CPU-TN checks, strict3 and quality gates
pass; final focused coverage of both owners100%, six frozen blobs match. Nine
SVGs and two PNGs inspected. ActualX3 and source syntheticT300 are separate.
The original default/pretty style demonstration is adapted to SVG/DOT, so Q-0741
remains open for that capability.23 closed/1257 open/393 active,1187 records;
next complete that style distinction using existing renderer ownership. No G3,
merge or repeated historical default timeout.

### Graphviz notebook style and signature groups (2026-11-12)

Q-0741 is accepted after implementation075aade6 completes its default/compact
views. The shared renderer retains geometry and accepts explicit signature-table
presentation. Original group order, local multidimensional labels, directional
row spans, compact headers and edge attributes match unchanged source tables.
Both styles retain every port and connection. All21 cells/13code execute, with
ten native SVGs and two composite-style PNGs inspected.

Thirty focused cases, including18 additional cases, pass. Coverage across three
owners is100% for141 statements/46 branches. The concentrated graph/typed-call
regression passes1713 tests in82.60seconds with eight parser warnings;84 CPU/TN
tests pass in16.80seconds with two warnings. Strict4, formatting, scope,
architecture, multi-level IR and protected API checks pass. Six frozen blobs
remain unchanged. ActualX3 and synthetic source modelT300 remain separate;
drawing adds zero gates, work qubits and measurements. Native Atom and boundary
lifetime naming are documented adaptations.

The ledger has24 closed/1256 open/392 active parents and1187 capability records.
Q-0742's generic foreign/subclass/typed drawing surface remains open. The next
batch audits Q-0743's complete source test file. An overly broad preliminary
selection was stopped and excluded from these acceptance totals. The historical
default timeout remains recorded. Evidence uses `g2-graph-styles-`.

### Complete Graphviz tests and independent kernels (2026-11-12)

Q-0743 is accepted through test commit7b0de0c0. All three source functions are
mapped. Four unchanged original cases execute, including the real notebook
helper and kernel. The original parameterized drawing assertion code object
also runs on native construction for both styles. Native declarations validate
seven unique DOT nodes, thirteen qualified endpoints and every edge reference.
The five source group-ID reservations are intentionally omitted by the native
representation and documented in GRAPHVIZ_NOTEBOOK.md.

The native notebook executes in a separate kernel with reference imports
refused, thirteen sequential code cells and ten SVG messages. Four new cases,
34 focused/214 Drawer regression/85 CPU-TN-notebook integration cases pass.
Focused coverage of the three existing owners remains100% (141 statements,
46 branches). Strict2, Black/Ruff/scope/architecture/IR/API checks pass. Eight
source blobs are unchanged. Initial loopback socket sandbox failures and their
successful permission-scoped reruns are retained. ActualX3, synthetic modelT300
and the zero-instruction drawing/test contribution remain separate.

The ledger has25 closed/1255 open/391 active parents and1187 capability records.
Q-0742's general drawing class/protocol scope remains open. The next batch
completes Q-0745's notebook container using the accepted score owners. Its exact
source blob7205fa53b168a8b0de5429345dc25d2081979446 has been restored and verified,
with seven cells/three code cells. Its relative JSON dump must run in a temporary
output directory to preserve frozen reference assets. Evidence uses
`g2-graph-source-`; the historical default-suite timeout remains recorded.


### Complete score browser assets (2026-11-13)

[Register score browser ownership](REGISTER_SCORE_ANIMATION.md) closes Q-0744
and Q-0746 after the complete HTML/JS, thirteen-function and global disposition.
Implementation de524ee0 adds manual domains through the shared renderer. The
original HTML loading order, source script and original serialized MultiAnd data
match native fixed graph geometry and computed styles. CDN relocation and the
clock/probe are explicit test-boundary instrumentation. Offline embedded data,
initial Adaptive style, transactional input admission and automatic fitting have
separate native dispositions. Safe domain magnitudes, positive canvas dimensions
and the 20,000-tick axis limit bound admitted browser work.

Twenty-five new cases bring the browser set to53 passes, alongside343 concentrated
and86 independent notebook/CPU/tensor passes without skips. Strict3, formatting,
scope, architecture, internal IR and protected API pass. Python exporter coverage
remains97%, while JS behavior has direct DOM assertions. Actual quantum primitives
and source analytical cost models remain separate. Presentation adds zero gates,
physical work or measurements.

The current ledger has29 closed/1251 open/387 active parents and1187 capability
records. Earlier totals above are dated historical batch snapshots. Q-0747's
Python protocol and Q-0748's original JSON asset remain open. The full default-suite
timeout remains recorded. G3 acceptance remains a separate stage.


### Context-aware score coordinates (2026-11-13)

Implementation `13f92bdf` extends the existing Drawer lane owner with source-like
call/register/element context through `new_y_for`. The default delegates to the
established no-argument `new_y`. Source method naming is adapted to preserve
existing subclasses. Scalar None and array tuple indices, reservations, empty
arrays, THRU retention, graph identity and full Add geometry match the unchanged
locked source through a replacement highest/lowest-lane policy.

Twenty-four new cases pass, with 367 concentrated and 86 independent integration
cases, zero skips. Strict2 and formatting/scope/architecture/IR/API pass. The lane
owner has 100% coverage of 57 statements and 22 branches. Evidence and the runnable
example are in REGISTER_SCORE_RESERVATIONS.md under `g2-score-coordinates-`.
There are zero added physical primitives, work wires or measurements. Actual
execution and source analytical accounts remain separate.

Q-0747 remains open for partial-line observations and general foreign protocols.
Counts remain 29 closed/1251 open/387 active parents and 1187 capability records.
The existing full-default timeout remains recorded in the ledger.


### Partial score lane lifecycle (2026-11-13)

Implementation `9b273636` ports open/close interval observations into the shared
Drawer lane manager. Immutable geometric segments are visible before and after
source-equivalent allocation, finish and release callbacks. Existing semantic
layout records supply register identities and word widths. The native cursor
integrates per-position source coordinates. Textual flavor values preserve
canonical quantum/classical and zero-width classical metadata.

The original source runs with a partial-line-dependent reservation policy and
custom coordinate policy. Add1–32 and scalar/vector/matrix/empty arrays match
all observations and complete geometry. Invalid-release preflight is an explicit
native adaptation, with the source's partial mutation reproduced in tests.
49 new/114 focused/416 concentrated/86 independent integration cases pass with
zero skips. Strict3 and formatting/scope/architecture/IR/API pass. Lane coverage
is 82 statements/26 branches at 100%, with combined lane/layout coverage97%.
Initial callback-keyword and Add(1) test-expectation failures are retained and
fixed. Evidence uses `g2-score-lifetimes-` and the runnable example is in
REGISTER_SCORE_RESERVATIONS.md.

Q-0747 remains open for general foreign/wire-symbol admission. Q-0748 retains
its independent original JSON asset boundary. Counts remain29 closed/1251 open/
387 active parents and1187 capability records. Added physical gates, work and
measurements are zero. Actual primitives and source models have separate
accounts. The existing full-default timeout remains recorded.


### Typed score register symbols (2026-11-13)

Implementation `34553f8d` extends existing Drawer score-data extraction with a
Core-register/index callback and recovers source predecessor/successor adjacency
ordering. Final root order comes from the existing layout. THRU output callbacks
run before their result is discarded. Stateful numbered glyphs and all callback
events match the unchanged source for Add1–32, shaped ports and reordered returns.
Existing consumers accept the resulting records unchanged.

32 new cases, 448 concentrated cases in 16.35 seconds and 86 independent
integration cases in 8.64 seconds pass with zero skips and eight/two existing
warnings. Strict3, formatting, scope, architecture, IR and API checks pass.
Score-data coverage is 98% across 143 statements and 68 branches. Evidence uses
`g2-score-register-symbols-`, with a runnable example and explicit native metadata
synthesis/final-order fallback in REGISTER_SCORE_SYMBOL_ADJOINT.md. The source
all-empty graph RightDangle error is reproduced alongside native empty output.
Initial test/typing failures are retained and corrected.

Q-0747 remains open for general foreign and record serialization boundaries.
Q-0748's original JSON asset and the historical default-suite timeout stay open.
Counts remain 29 closed, 1251 open, 387 active parents and 1187 capability records.
Added physical gates, work and measurements are zero. Existing amplitude,
all-branch work restoration and append regressions pass. Physical primitives
and source analytical models remain separately reported.


### Partial lane record serialization (2026-11-13)

Implementation `329e0a64` adds source-format json_dict to existing immutable
LaneSegment. Open endpoints export null, closed endpoints retain their integer
coordinates and both flavor strings match the source. Each mapping owns its
values. Native mapping conversion uses the standard JSON encoder. Complete score
records retain their existing integer-endpoint and UTF-8/newline policies.
Source RegPosition, SoqData, VLine, all six glyph classes and complete score
records are directly compared through the original encoder. Scalar/vector/matrix/
empty arrays and quantum/classical/zero-width release snapshots round-trip.

40 new/focused cases in 1.24 seconds, 488 concentrated cases in 16.89 seconds and
86 independent integration cases in 8.97 seconds pass with zero skips and eight/
two existing warnings. Strict2, formatting, scope, architecture, IR and API checks
pass. Lane coverage is 84 statements and 26 branches at 100%. The initial test
hash transcription failure is corrected and retained. Evidence uses
`g2-score-records-`, with a runnable example in REGISTER_SCORE_RESERVATIONS.md.

Q-0747 remains open for general foreign graph/dtype/manager/glyph admission.
Q-0748's original JSON asset and historical default-suite timeout stay open.
Counts remain 29 closed, 1251 open, 387 active parents and 1187 capability records.
Added physical gates, work and measurements are zero. Existing amplitude,
all-branch work restoration and append regressions pass. Physical primitives
and source analytical models remain separately reported.


### Native custom port painters (2026-11-13)

Implementation `1247d5fe` exposes the existing builtin port painter for reuse and
adds an optional renderer-local replacement callback. Record-order invocation,
positive row coordinates, boundary ports, builtin delegation and false-valued
callables are accepted. A custom source WireSymbol subclass with portable TextBox
fallback matches every native Agg pixel at three size limits. Source generic
custom adjoint identity is preserved. Custom inverse presentation stays supplier-
owned. Callback failures close incomplete native figures, with the source's
retained figure behavior directly reproduced and explicitly adapted.

15 new cases, 503 concentrated cases in 16.72 seconds and 86 independent
integration cases in 9.14 seconds pass with zero skips and eight/two existing
warnings. Strict2, formatting, scope, architecture, IR and API checks pass.
Matplotlib score coverage is 60 statements and 18 branches at 100%. The initial
strict failure at unannotated test-only Agg methods is corrected and retained.
The PNG was inspected. Evidence uses `g2-score-custom-symbols-`, with a runnable
example in REGISTER_SCORE_SYMBOL_ADJOINT.md and a Drawer README entry.

Portable JSON uses the existing six glyph classes, with the custom painter
supplied again for local appearance. Q-0747 remains open for foreign admission,
arbitrary serialized symbol names and custom connector-label drawing. Q-0748's
original JSON asset and historical default-suite timeout stay open. Counts remain
29 closed, 1251 open, 387 active parents and 1187 capability records. Added
physical gates, work and measurements are zero. Existing amplitudes, branch
work restoration and append regressions pass. Actual primitives and source
analytical estimates retain separate accounts.


### Native connector label painters (2026-11-13)

Implementation `c1f92dc7` adds label_drawer over the existing ScoreVerticalLine
and Text label mapping. Shared builtin label drawing remains in the Matplotlib
owner. Each nonempty label follows its connector, before all ports. Source
half-row coordinates, whitespace/empty admission and false-valued callables are
accepted. Custom source Text subclass dispatch matches independent native Agg
pixels at three size limits and bottom rows zero/two/five. The full event list
compares labels, ports and already-drawn line collections. The PNG was inspected.

11 new/focused cases in 1.92 seconds, 514 concentrated cases in 19.09 seconds
and 86 independent integration cases in 9.58 seconds pass with zero skips and
eight/two existing warnings. Strict2, formatting, scope, architecture, IR and API
checks pass. Matplotlib score coverage is 63 statements and 18 branches at 100%.
The initial observer-fixture serialized-class failure is corrected by its
explicit portable Text fallback and retained. Source files are unchanged.
Evidence uses `g2-score-custom-labels-`, with an executable example in
REGISTER_SCORE_SYMBOL_ADJOINT.md and a Drawer README entry.

RuntimeError and KeyboardInterrupt share native incomplete-figure cleanup and
preserve exception identity. The source retained-figure behavior is reproduced.
Retry and empty-graph ownership pass. Custom appearance uses the local painter
again after loading portable Text records. Q-0747 remains open for general foreign
and arbitrary serialized symbol-class admission. Q-0748's original JSON asset
and historical default-suite timeout stay open. Counts remain 29 closed,
1251 open, 387 active parents and 1187 capability records. Added physical gates,
work and measurements are zero. Existing amplitude, all-branch work restoration
and append regressions pass. Actual primitives and source models stay separate.


### Original score JSON asset (2026-11-13)

Q-0748 closes with `11facf75`. The unmodified 4,681-byte fixture has original
Git blob b5ed08cad72ae69d275fc89b90337b7dc4daf3a8, distinct from generated
6253c8e4b4405560df866d6dd2581ba0b3f9cd4f outputs. Source attribution is retained.
Every field survives sequential export and HTTP import. The dependency view has
its explicit column projection. Frozen HTML relative loading, fixed SVG geometry,
fonts, axes and outlines compare directly in real Chrome. The completed screenshot
was inspected. Source JS does not paint connector-label text. Native page/control,
adaptive-spacing and Matplotlib record boundaries remain explicit.

Four new cases, 55 browser cases in 55.48 seconds, 516 concentrated cases in
23.52 seconds and 86 integration cases in 8.44 seconds pass with zero skips.
Existing warning counts are eight and two. Strict3/Black/Ruff/scope/architecture/
IR/API checks pass. HTML owner has 46 statements and 12/14 branches covered,
97% combined, with DOM assertions for JS. The initial dependency-column
expectation failure was corrected and retained. Evidence uses g2-score-asset-.

The ledger now has 30 closed, 1250 open and 386 active parents. Capability
records remain 1187, extending Q-0748.01. Q-0747 general foreign/serialized-symbol
admission and the historical 600-second default-suite timeout remain open.
The asset adds zero physical gates, work or measurements. Existing gate-order,
complex-amplitude, all-branch work restoration and append regressions pass.
Actual primitive and source-model costs remain in their separate resource table.


### Mixed storage register scores (2026-11-13)

Implementation a8729432 shares the existing lane storage classifier with completed
RegisterLine lifetimes, retaining a default quantum flavor for hand-authored
records. Core-owned CBit and zero-width quantum components reach the shared
layout and data exporter. Logical scalar lanes survive width zero. Empty arrays
have no lanes. Classical integer metadata uses direct layout while its Core
execution refusal is tested. The existing explicit classical override survives.
A measured parity/discard graph verifies quantum-to-classical reuse and identical
physical binding before and after drawing.

31 new cases and 52 focused cases pass. Final regression passes 583 cases in
19.37 seconds, integration 86 in 25.55 seconds and browser 14 in 20.95 seconds,
with zero skips and eight/two existing warnings. Strict6/Black/Ruff/scope/
architecture/IR/API pass. Four owners cover 457/463 statements and 173/180
branches, 98% combined. Lane classification/heap covers all 87 statements and
26 branches. The mixed PNG was inspected. Reference code remains unmodified.

Source array, geometry, glyph and callback comparisons pass. Pixel equality
explicitly adapts boundary text and port-record order. Long overlapping label
paint order is a recorded open Q-0747 producer gap. General foreign and arbitrary
serialized-symbol admission remain open. Initial executable classical-integer
fixture refusal, unadapted pixel-order mismatch and root-signature fixture
failure are retained with corrected acceptance boundaries. Classical metadata
execution admission and source reference files are unchanged. Evidence uses
g2-score-mixed-. Existing all-branch complex-amplitude, parity/discard, work-zero
and append regressions pass. The measured witness emits H1, measurement1 and
classical_xor1, while the source MeasureX model records measurement1 and Clifford0.
Parity has no locked analytical model. Drawing adds zero gates/work/measurements.
Counts remain 30 closed, 1250 open, 386 active and 1187 capability records.
The previous full-default600s timeout remains open. Next inspect source port
traversal and long-label overdraw in the existing record producer.


### G2 score adjacency traversal follow-up

Implementation `0054242d` reuses the existing neighbor-group reconstruction for
LeftDangle, component instances and RightDangle. It preserves callback cadence
and native identities while correcting producer overdraw order. Twenty mixed
Agg cases compare directly without source port reordering. Boundary text remains
an explicit native naming adaptation. Add 1–32-bit ordered records, idle wires,
empty graphs and swapped returns through actual source finalization pass.
Hand-authored scores retain completed-line fallback. Arbitrary mutated source
edge insertion histories remain outside native graph admission.

The batch adds 24 cases and passes 87 focused, 607 concentrated, 86 independent
integration and 14 browser cases with zero skips. Strict seven-file typing,
format, Compiler scope, architecture, IR and public API checks pass. Four owners
cover 464/470 statements and 173/180 branches, 98% combined. Old producer negative
control fails as expected. Source bytes remain unchanged. Prior physical counts,
all-branch amplitudes, clean work and append regressions pass. Display adds zero
physical gates, work or measurements, and measured/model accounts remain separate.
Evidence uses g2-score-order-. Q-0747 general foreign/serialized-symbol admission
remains open. Counts remain 30 accepted, 1250 open, 386 active parents and 1187
capability records. This extends Q-0747.05 without inferring whole-parent closure.


### G2 default register-symbol follow-up

Implementation `7f980ce1` reuses existing Core register groups and the Drawer
symbol selector for default declaration/index labels and source title truncation.
Native function names own titles. Existing callbacks and gate styles retain
precedence, while plain metadata-free layouts retain their identity labels.
Fifty-one new cases execute the unchanged upstream default symbol method,
including three full Agg comparisons with only boundary text adapted. Frozen
Bloq and musical-score blobs match the inventory. Empty-source RightDangle
failure and native empty-score behavior remain explicit.

Local acceptance passes 51 focused, 658 concentrated, 86 independent integration
and 14 browser cases without skips. Strict two-file typing, format, Compiler
scope, architecture, IR and public API checks pass. Four owners cover 468/474
statements and 175/182 branches, 98% combined. Baseline negative control fails
as expected. Physical instructions, all-branch amplitudes, clean work and append
regressions pass. Drawing adds zero gates, work or measurements. Measured and
analytical resource accounts remain separate. Evidence uses g2-score-default-.
Q-0747 remains open for general foreign/serialized-symbol admission. Counts
remain 30 accepted, 1250 open, 386 active parents and 1187 capability records.


### G2 connection-seeded score traversal follow-up

Implementation `27be2062` shares connection encounter order between root lane
placement and ordered symbol records. Input/output boundaries participate in
source topological traversal. The reverse-root baseline and idle input boundary
failures expose the previously uncovered cases. Frozen source remains unchanged.
Acceptance passes 130 focused, 788 concentrated, 86 independent integration and
14 browser cases without skips, plus strict three-file typing, formatting,
Compiler scope, architecture, IR and protected API checks. Four owners cover
486/492 statements and 185/192 branches, 98% combined. Full amplitudes, all
observed measurement branches, clean work, primitive instructions and ordinary
append regressions pass. Drawing adds zero gates/work/measurements, while actual
primitive counts and analytical source estimates remain separately reported.
Evidence uses g2-score-roots-. Native names/boundary text and isolated zero-port
insertion order are explicit adaptations. General foreign/serialized-symbol
admission keeps Q-0747 open. Counts remain 30 accepted, 1250 open, 386 active
parents and 1187 capability records.


### G2 resource flame data

Implementation `b7111c18` adds source-ordered folded T-cost DFS through Drawer.
Existing Compiler graph and cost owners supply keys, weights and full costs.
The generator repeats complete child subtrees, retains positive-successor
filtering and explicit T/T-adjoint collapse, and supports bounded partial reads
at a trillion repetitions. Native title and concrete-integer admission are
explicit adaptations. All eight source definitions have dispositions in
RESOURCE_FLAME_GRAPH.md. Automatic source naming/keep recognition, the convenience
facade and interactive SVG remain pending in Q-0739.

Acceptance passes 69 focused, 857 concentrated, 51 cost/Fredkin and 86 independent
integration cases without skips. Strict two-file typing, formatting, Compiler
scope, architecture, internal IR and API checks pass. The new owner covers all
28 statements and 16 branches. The four frozen source blobs match inventory.
All four original test functions retain their assertions, including the QFT
historical expectation failure for both producers. Actual QFT data agrees.
The original source file reports three passes and one xfail. Model/visible-graph
cost mismatch behavior is recorded alongside separate physical and modeled
resource counts. Drawing adds zero gates/work/measurements. Counts remain 30
accepted and 1250 open, with 388 active and 1189 capability records. Q-0740 remains
open for its QFT expectation disposition. Evidence uses g2-flame-data-.


### G2 resource flame convenience and rendering

Compiler query_terminal_t_cost reuses existing resource declarations, recursive
caches and terminal precision costs. Drawer iter_resource_flame_graph_data
composes full costs, visibility depth, native titles and source-equivalent leaf
recognition for ordered roots. render_flame_graph_svg streams UTF-8 data to an
explicit executable, closes input before execution and removes temporary state
on success, timeout, process failure and generator failure.

Acceptance adds 53 cases. The four source examples match rows with four depth
cutoffs and both keep policies. Actual CSwap SVG matches untouched source output
after normalizing only Perl hash-dependent nonoverlapping frame order. Three
reference asset blobs match inventory. There are 961 concentrated and 86
independent integration passes without skips. Strict four-file typing, format,
Compiler scope, architecture, IR and API checks pass. Drawer flame coverage is
73 statements and 24 branches at 100 percent. Combined terminal/Drawer coverage
is 98 percent. A native 94-T SVG example executes.

Native titles, concrete rendering admission, caller-owned persistent files and
explicit executables are recorded adaptations. Exclusive T-bearing aggregates
require explicit callee declarations. Browser file-protocol preview was blocked,
so interactive inspection remains open alongside foreign name reflection,
asset distribution and the QFT historical expectation discrepancy. Q-0739.02
and Q-0739.03 are accepted subitems. Whole-source parents remain open. Counts
are 30 accepted, 1250 open, 388 active and 1191 capability records, including one
historical record. Resource evidence separates primitive Fredkin CX8/H2/T4/Tdg3
from terminal7T/QEC4T models, with zero added drawing gates/work/measurements.
Evidence uses g2-flame-composed- and g2-flame-workflow-.


### G2 automatic compound resource costing

Implementation ef820936 replaces the QEC-local callee implementation with
Compiler resource_decomposition. QEC costing and precise terminal/display
traversal share the existing recipes, preserving separate terminal boundaries.
Automatic swap, conjunction, controlled phase/rotation and predicate roots now
compute their full costs without copied source declarations. Negative-baseline
execution proves prior zeros for CSwap10, CZPow0.125/eps1e-4 and MultiAnd4,
corrected to source-equivalent 70T, 29T and 12T.

Acceptance adds 148 tests. Seventeen source root pairs times four cutoffs and
two keep policies give 136 exact row comparisons. Seven source blobs match.
There are 1151 concentrated and 86 independent integration passes with zero
skips. Strict six-file typing, format, Compiler scope, architecture, IR and API
checks pass. Shared recipes cover 68 statements and 46 branches at 100 percent.
QEC and Drawer owners also cover 100 percent, with combined four-owner coverage
99 percent. Existing exact primitive, merged-control measurement branches,
complex amplitudes, work-zero and append checks pass. Terminal Fredkin7T,
QEC default4T and actual CX8/H2/T4/Tdg3 remain separate, while merged Fredkin
emits11T/measure1 and its terminal model gives4+7+0.

This extends Q-0739 with accepted child Q-0739.04. Whole-file parents remain
open, with 30 accepted, 1250 open, 388 active and 1192 capability records,
including one historical record. Foreign names, renderer distribution requiring
protected packaging changes, browser interaction, historical QFT expectations
and general foreign admission remain explicit gaps. Evidence uses g2-flame-auto-.

## G2 concise resource flame labels

Implementation dcd3011f completes the default native naming path through the
existing resource dataclasses and Drawer title callback. Source tuple, ndarray,
float and nested-name rules are exercised directly against frozen helpers.
Native class and constructor names remain an explicit presentation adaptation.
Component names and custom strings retain precedence. Generated arithmetic
fields are excluded. Colliding compact titles retain distinct resource keys.

Acceptance adds 61 cases, with 106 focused, 1212 concentrated and 86 independent
integration passes, zero skips, strict typing of three files, format, scope,
architecture, IR and protected API checks. Drawer coverage is 93 statements and
38 branches at 100 percent. Combined resource owner coverage is 99 percent.
Two source blobs, the exact prior-head verbose-label witness and four executable
documentation examples pass. Evidence uses g2-flame-label-.

Drawing adds zero primitive gates, work wires or measurements. The concentrated
suite retains exact Fredkin complex-amplitude and measured merged-Fredkin
branch/work cleanup verification with physical counts and model estimates in
separate accounting. The source metadata fixtures test reflection only.
Sixteen actual source CSwap root/depth paths compare ordered rows after explicit
native name substitution at the reference boundary.

Q-0739.05 is accepted. Parent counts are 30 accepted, 1250 open and 388 active,
with 1193 capability records, comprising 1192 current and one historical.
Q-0739 retains foreign admission, licensed renderer distribution and interactive
inspection. The historical QFT assertion and full-default timeout remain open.
The next asset is Q-0738, the call-graph drawing notebook, using existing owners.

## G2 complete resource call-graph notebook

Implementation 303b4c33 integrates the seven-cell, three-code-cell source asset
through native symbolic graph, QEC cache and existing Drawer DOT/SVG owners.
All original and native code cells execute independently. Nine nodes and nine
edges, symbolic multiplicities, full leaf totals, per-node cost annotations,
five depth settings and title/annotation customization are compared.
A real independent kernel executes the native notebook and validates both
SVG displays. The byte-export cell retains its third SVG. Native imports reject
reference SDK and graph-library dependencies during standalone cell execution.

Thirteen focused, 1225 concentrated and 87 integration cases pass with zero
skips. Strict typing of three files, format, team scope, architecture, IR and
protected API checks pass. New example coverage is 31 statements and two
branches at 100 percent. Existing Drawer coverage is 92 percent, combined
93 percent. Five source blobs remain unchanged, including the notebook restored
from its exact Git object. The native rendered PNG was inspected locally.

Native declaration names, existing table layout and explicit title/annotation
callbacks are the presentation adaptations. The source demonstration's atom
static100T and numerical X tensor remain separate accounting. Drawing emits
zero quantum gates, work wires or measurements. Existing measured branch and
work-zero regressions remain in the concentrated run. CALL_GRAPH_NOTEBOOK.md
records the complete source cell mapping and retained initial verification logs.

Q-0738 is closed with Q-0738.01. Whole-parent counts are 31 accepted, 1249 open
and 388 active. There are 1194 capability records, comprising 1193 current and
one historical. Q-0734/Q-0735 and the generic drawing facade retain separate
scope. The next module is the whole-source Q-0735 test-asset audit.
