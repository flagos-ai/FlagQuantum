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
(Q-0880, Q-0886, Q-0890, Q-0896), bring the total to **17 closed files and 1,263 open files**. These file
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
