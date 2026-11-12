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
