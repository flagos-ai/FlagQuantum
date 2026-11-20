# Arithmetic multiplication file audit

Q-0207, Q-0208 and Q-0209 cover `arithmetic/multiplication.ipynb`,
`multiplication.py` and `multiplication_test.py` at Qualtran v0.7.0
commit `bcb17bb25360ed68ce3bc084645db3e929799389`. The original three
Git blobs are verified against their frozen identities. The batch reuses the
accepted execution owners and adds native teaching assets and declaration
conformance for all eight original factories.

## Ownership and contracts

Compiler owns big-endian word lowering, logical result binding and caller-owned
workspace. The shared partial product, controlled add, binary column
compression, control and subtraction owners supply the direct dependencies.
Simulation owns the original square and product accumulation value callbacks.
The examples keep source port metadata and declared costs and use ordinary
`circuit.append(op)` for the executable path. These are single-device checks.
The primitive branch tests verify reset of work and feedback, along with
complex output states. RIGHT ports of the original declarations describe
logical outputs. Native compute/erase uses explicitly bound physical words.

| Source class and original definitions | Owner and retained evidence |
|---|---|
| `PlusEqualProduct`: attrs validation, signature and three dtype properties, adjoint, apply/classical callback, tensors, diagram/wire labels, register shape and rebinding, call graph | Arithmetic `product_accumulation` and Simulation `classical_product_accumulation`. Native forward/adjoint preserve wraparound and exact phase. Source tensor and basis behavior, complete small histories and original numeric overflow/broadcast behavior are already covered. The original Cirq protocol and generic rebinding remain foreign boundaries. |
| `Square`: signature, classical/tensor callbacks, wire label, graph, adjoint | Arithmetic `integer_square` and Simulation `classical_square`. The RIGHT result becomes LEFT for erasure in the source. Both callbacks and complete complex trajectories are already covered. Compiler uses the paper's compression procedure. |
| `SumOfSquares`: signature, label, graph | Arithmetic `sum_of_squares` and shared arithmetic cost model. Retains array shape, unsigned output width and the count divisible-by-three adjustment. The independent integer oracle and prior branch proof cover native execution. |
| `Product`: signature, label, graph | Arithmetic `integer_product` and shared cost model. Keeps the twice-maximum result width and paper partial-product construction. Native execution is checked by the zero-target accumulation contract. |
| `ScaleIntByReal`: signature, label, graph | Fixed-point scaling owner. Retains mixed output format and the original cost. Native D4 truncation and controlled add/erase are covered separately. |
| `MultiplyTwoReals`: signature, label, graph | Fixed-point product owner. Retains the equal pure-fraction formats and D13 estimate. Native D5 cutoffs and small branch cleanup are covered. A one-bit source estimate is negative and native emits zero gates. |
| `SquareRealNumber`: attrs validation, signature, label, graph | Fixed-point square owner. The declaration retains both input ports. Native D24/D28 uses one source word. Existing tests cover the truncation, clean work and zero result edge. |
| `InvertRealNumber`: attrs validation, signature, label, graph | Fixed-point reciprocal seed, Newton step and orchestration owners. Source uses mixed THRU output and declares all eight child kinds, including positive-polarity multi-control X. Native Fig. 4 seed, multiply/subtract with widened mixed inputs, and pure-fraction result cleanup have separate evidence. The source cost graph rejects widths one and two through its fixed-point square child. |

The complete original 66-cell notebook covers eight factories and their
documentation specifications. The native notebook adds one execution
section, retaining the original equations, examples, signature topics, and
one-level call edges and leaf counts. Graph names, tables and labels use
native presentation. The fixed-point square and reciprocal sections distinguish
source metadata from executable contracts.

## Original test disposition

The eight autotesters cover original factories, with three skipped modes per
factory. Each class lacks a declared decomposition, so decomposition, equivalent
decomposition counts and dtype consistency checks skip at the source. The eight
direct tests exercise the square tensor/classical values and reversal, sum of
squares width and resource count, product/scaling/fixed-point product/fixed-point
square resource counts, accumulation basis permutation and estimate, and the
complete reciprocal cost composition. All 24 runnable non-notebook source
cases pass. The seventeenth function is the notebook check, exercised as an
independent kernel against the immutable notebook. The native catalogue checks
all eight signatures, fractional formats, and complete direct call graphs
and leaf counts against the original factories.

## Resource accounts and closure

| Scenario | Native emitted primitives and workspace | Declared source model |
|---|---|---|
| Eight-bit integer square | 224 T, 575 CX, 56 measurements, 56 clean work bits | 56 Toffoli |
| Four-by-six integer product | 168 T, 410 CX, 42 measurements, 42 clean work bits | 42 Toffoli |
| Three two-bit words summed as squares | 32 T, 95 CX, 8 measurements, 8 clean work bits | 9 Toffoli |
| Two-by-two product accumulation into four bits | 36 T, 73 CX, 9 measurements, 9 clean work bits | 32 T approximation |

These counts reuse prior branch and gate evidence. Primitive T gates and
source Toffoli-equivalent estimates are separate quantities. The source
operations lack gate-level expansion, so exact primitive order against
those roots remains unavailable. Prior tests compare the supported source
callbacks, signatures, paper-defined dependencies, complete complex states
and small measurement histories. No newly emitted execution stream is
introduced here.

Q-0207 is a complete teaching asset. Q-0208 and Q-0209 remain open for
missing original decomposition and its autotester modes, generic foreign
construction, diagram callbacks, rebinding and serialization, and the
fixed-point square and reciprocal contract differences documented above.

## Stored Square direct dependency, November 20, 2026

`arithmetic/integer_square_component.py` now composes the existing square through
`bits/weighted_columns_component.py`. It preserves THRU unsigned `a`, RIGHT
result creation and matching LEFT erasure. The column graph owns a fixed clean
work block and uses the shared measured And primitive calls. Both directions
keep source coherence and free restored owned storage. Contiguous indexing is
its first nested consumer, with an append test replacing the flat square root.
The full evidence and independent model/primitive accounts are recorded in
CONTIGUOUS_INDEX_FILE_AUDIT.md. The unchanged multiplication file's eight-class
inventory and original test evidence above are reused. Product, sum-of-squares
and fixed-point stored composition remain distinct open work. This addition
leaves Q-0208/Q-0209 open and makes no upstream square-decomposition claim.

## Stored Product and SumOfSquares dependency, November 20, 2026

`integer_product_component.py` preserves the source's THRU unsigned a/b words
and RIGHT result of twice the larger input width. Its inverse consumes the
matching LEFT product image. `sum_of_squares_component.py` preserves the
row-major shaped THRU input and RIGHT result width
`2 * width + (count - 1).bit_length()`. Its inverse consumes the matching image.
Both use `column_result_component.py`, also used by Square. This owner handles
only result allocation, typed Split/Join views and restoring input aliases.
The existing weighted-column owner retains products, carries, measured And
calls and scratch erasure. Physical binding flattens shaped ports row-major.
Inverse callers transfer result ownership through `input_ownership`.

This replaces duplicated Square composition. Twenty-four complete bound
Square records, widths 1 through 12 in both directions, are exactly equal to
base `e7ea0499d`, including instructions, aliases, feedback slots and workspace
events. The first comparison detected a changed zero-gate view order before
inverse result release. Restoring output Join before input Join preserves the
original release positions. The final comparison passes without normalization.

Stored and expanded Product/SumOfSquares primitive multisets, per-wire dependency
order, operands and feedback match their accepted flat owners. Measurement SSA
slots are renamed only after verifying each feedback reads its actual preceding
measurement. Independent gates can have different global schedules. Existing
unchanged And leaf comparisons and complete channel certificates are reused.
The original Product and SumOfSquares roots supply signatures and models, with
no executable decomposition, tensor or classical callback. This addition uses
the already documented paper construction. It supplies no original-root
primitive-order comparison.

Complete coherent small inputs, entangled spectators and zero scratch are
checked in both directions. Product uses widths 1x1 and 2x2. SumOfSquares uses
three one-bit words and two two-bit words. The one-AND cases exhaust all four
forward/inverse measurement history pairs. Larger cases use two trajectory
seeds and reuse the accepted leaf/channel evidence. Normal `circuit.append(op)`
round trips pass with the flat roots replaced by failing stubs. Shapes, inverse
ownership, stored/expanded calls and invalid arguments are checked separately.

| Stored operation | Observed primitives and owned storage | Original analytical model |
| --- | --- | --- |
| Product 4x6 | 168 T/Tdg, 410 CX, 42 measurements, 42 scratch plus 12 output | 42 Toffoli |
| SumOfSquares three two-bit words | 32 T/Tdg, 95 CX, 8 measurements, 8 scratch plus 6 output | 9 Toffoli |

These streams retain the flat implementation's physical counts. The sum model
and actual measured construction differ and remain separately reported.
`g2-column-results-evidence.json` records six scenarios, five immutable source
blobs, Qualtran 0.7.0 and Kickmix 0.1 Simulator presence, and the 24 exact Square
comparisons. Strict typing passes all five Python files. Black, Ruff,
Integration scope, architecture, IR and public API checks pass. Dependency policy
reports 687 existing whole-repository findings and zero batch-path findings.
Concentrated regression passes 1,400 checks with one existing IPython warning
and no skips. Local integration passes 17 checks. Final focused checks pass
36 cases. Receipts use `g2-column-results-`. The earlier
regression interrupted by the user is retained separately and excluded from
acceptance. The known default docstring coverage failure is reused after
verifying its ten implicated files against base `e7ea0499d`. This supplies no
clean-default-suite claim. The unchanged eight-class catalogue and original 24 runnable cases
and skipped decomposition modes remain reusable evidence.

Q-0208/Q-0209 remain open for the fixed-point and reciprocal executable contract
differences and generic foreign construction, drawing, rebinding and
serialization recorded above. The next closure must inventory those complete
remaining definitions before introducing further arithmetic dependencies.

## Approved unsigned fixed-point storage, November 20, 2026

Proposal 115 closes the physical storage admission dependency of the four
fixed-point signatures. Canonical multi-bit unsigned QFxp words now preserve
nominal type, fractional metadata, MSB raw bits, shaped views and owned
RIGHT/LEFT storage through physical binding. Shared Add and its exact inverse
execute with full complex-state, all small-history and zero-work checks.
Compiler retains wider fixed-point predicate refusal explicitly. See
UNSIGNED_FIXED_POINT_STORAGE_AUDIT.md for evidence and separate observed/model
counts. This dependency leaves SquareRealNumber's unused b port,
InvertRealNumber's mixed THRU output scaling and the remaining generic foreign
interfaces open. Original missing root decompositions remain recorded.

## Stored SquareRealNumber dependency, November 20, 2026

`fixed_point/fixed_point_square_component.py` closes the typed signature and
owned-result dependency of SquareRealNumber. Its original two pure-fraction
THRU inputs a and b both survive binding. The existing D24/D28 construction
reads only a. The returned fixed-point result is RIGHT for compute and LEFT for
matching image erasure. Shared column_result now accepts an explicit result
descriptor, used by this fourth consumer. It retains the original uint default
for the three integer consumers. The weighted-column owner retains all pair
products, carry storage, measured And calls and zero-work release.

Stored and expanded calls match the accepted flat paper construction's primitive
multisets, per-wire order, operands and validated feedback-slot renaming at
widths 3, 4, 5, 6, 8 and 10, in both directions. No gate addresses b. The original
root provides its dual-input signature and analytical formula without an
executable decomposition. Original leaf/channel evidence is reused. Independent
gates can have different global schedules.

Complete coherent a/b inputs, entangled spectator and zero scratch pass at
widths three and four, including both inverse directions. Width three exhausts
all sixteen forward/inverse measurement history pairs. Width four uses two
trajectory seeds and the accepted complete leaf/channel certificates. Ordinary
circuit.append(op) compute/erase passes with the flat root replaced by a failing
stub. The odd/even truncation oracle is reused from the accepted flat module.

Source models and observed counts remain separate. At width three, the native
paper construction emits 8 T/Tdg, two measurements and two scratch bits, while
the source model reports zero Toffoli. At width four it emits 28 T/Tdg, seven
measurements and seven scratch bits, versus source four Toffoli. The discrepancy
was already recorded for the flat executor and survives typed composition.

The remaining fixed-point executable definitions are ScaleIntByReal and
MultiplyTwoReals typed truncated call graphs, and InvertRealNumber's mixed THRU
result contract. Generic foreign construction, drawing, rebinding and
serialization retain their existing disposition. Q-0208 and Q-0209 remain open.

Stored-square receipts use `ftqc-reference-env/g2-fixed-square-stored-`.
Concentrated regression passes 3,170 checks with one existing IPython warning
and zero skips. Local integration passes 17 checks. Final focused checks pass
66 cases. Strict typing covers three files. Black, Ruff, scope, architecture,
IR and public API checks pass. Four immutable source blobs, Qualtran 0.7.0 and
Kickmix 0.1 Simulator are verified. Dependency policy reports 687 existing
whole-repository findings and zero touched-path findings.

Thirty-six complete bound integer Square/Product/SumOfSquares records match
base d6b5b2448 exactly in both directions, including gates, aliases, measurement
slots and allocation/release events. The comparison loads the original shared
helper as well as the original three callers, so the extended helper itself is
covered. Default docstring coverage failure evidence is reused after verifying
the ten implicated files against that base. Default-suite completion remains
open. Type-only test annotations were corrected after broad collection and are
included in the final focused run.

## Stored ScaleIntByReal and MultiplyTwoReals, November 20, 2026

The two typed roots close the D4/D5 stored-call dependencies through
`fixed_point/truncated_product_component.py`. ScaleIntByReal retains pure-fraction
THRU real_in, unsigned THRU int_in and a RIGHT result with real_width-integer_width
fractional bits. MultiplyTwoReals retains pure-fraction a, b and result. The exact
inverse consumes a matching LEFT result image and releases zeroed storage.
All root words retain original nominal descriptors and MSB-first order.

The common owner uses the existing initial And compute/measurement erase
primitives and imports the complete accepted controlled_addition_program for
each growing prefix/suffix pair. CAdd retains its original And/Add/views and
logical exact adjoint. Result storage owns the initial copied conjunctions.
Each CAdd owns its loaded word and carries until their measured release. The
flat D4/D5 owners, CAdd/Add owners and their accepted source-parity tests remain
byte-identical to base c5271cbec. The frozen roots have signatures and resource
models without executable decompositions.

Original CAdd/adjoint decompositions are compared on every retained step with
the actual data/control/result operands and native child allocation lifetimes.
Fresh classical slots normalize only after checking the measurement supplying
each feedback. Primitive multisets, per-wire order and conditional operands
match those original child streams. Initial And leaf evidence is reused. Whole
root primitive multisets match the unchanged accepted flat D4/D5 emitters.
Independent global scheduling can differ.

Each CAdd now reuses its own scratch lifetimes, so its physical temporary names
can differ from the flat emitter's two fixed pools. Expanding calls can also
move RIGHT result allocation behind an independent temporary allocation. The
tests check live-owner disjointness, pool membership, returned output mapping,
and clean release. Complete complex-state checks use the actual mapping in
both stored and expanded paths. Caller input aliases remain identical.

Full coherent inputs and an entangled spectator are checked for scaling 2x1,
2x2 and 3x2, and fractional multiplication widths one, two and three. Forward
and inverse restore all work. Widths at most two exhaust every measurement
history pair, including sixteen pairs for scaling 2x2. Larger cases use two
trajectory seeds and reuse the accepted complete And-channel certificates.
Normal circuit.append(op) round trips pass with both flat roots replaced by
failing stubs. The one-bit fractional result is zero, while its unchanged source
model reports negative one Toffoli and negative four T-equivalent.

Resource receipts separate actual primitive counts, peak owned wires including
the result, total allocated wire lifetimes and original forward models. Dynamic
wire reuse therefore retains both capacity and lifetime counts. The remaining
whole-file executable contract is InvertRealNumber's mixed THRU result, alongside
PlusEqualProduct's typed accumulation and generic foreign definitions. The
source roots' missing decompositions retain their existing disposition.
Q-0208 and Q-0209 remain open until their full remaining scope is disposed.

| Stored product | Observed forward primitives and owned storage | Original forward model |
| --- | --- | --- |
| ScaleIntByReal 6x4 | 104 T/Tdg, 216 CX, 24 measurements/resets; 10 scratch plus 6 output peak, 30 allocated wire lifetimes | 26 Toffoli, 104 T-equivalent |
| MultiplyTwoReals 4 bits | 44 T/Tdg, 88 CX, 10 measurements/resets; 6 scratch plus 4 output peak, 14 allocated wire lifetimes | 11 Toffoli, 44 T-equivalent |

The inverse of these examples emits 96 and 40 T/Tdg respectively, with 26 and
11 measurements/resets. The original forward model is kept in its own column.
The initial copied conjunctions account for the forward/inverse asymmetry.

Receipts use `ftqc-reference-env/g2-truncated-products-`. Concentrated regression
passes 3,962 checks with one existing IPython warning and zero skips. Local
integration passes 17 checks. Final focused acceptance passes 45 cases, including
nominal-descriptor assertions added after broad collection. Strict typing covers
four Python files. Black, Ruff, scope, architecture, IR and public API checks
pass. Five original source blobs, Qualtran 0.7.0 and Kickmix 0.1 Simulator are
verified. Dependency policy reports 687 existing whole-repository findings and
zero touched-path findings. Default docstring coverage failure is reused after
verifying all ten implicated files against c5271cbec. Default-suite completion
remains open. Earlier expanded-output coordinate assumptions were corrected in
the tests, then both stored and expanded coherent execution were verified.

### Negative fractional scaling outputs

The complete source signature also accepts integer_width greater than
real_width. Proposal 115 already admits the resulting negative fractional
metadata. The stored scaling constructor now covers every pair of positive
concrete input widths. For a negative initial retained width, it discards
integer_width-real_width+1 low multiplier bits whose terms shift every source
bit past the result cutoff. The first remaining term uses a one-bit And copy;
subsequent terms reuse the same growing CAdd calls. A one-bit real word has no
surviving term in this range and returns a zero image.

The independent quantization oracle remains the accepted D4 sum of individually
truncated terms. Native output raw bits represent that sum in the original
QFxp(real_width, real_width-integer_width) format. Both original input words
survive, including ignored low multiplier bits. The inverse clears the computed
image. Stored/expanded source-CAdd comparison covers 1x3, 2x3, 2x4, 3x4 and 3x5.
Complete coherent 2x3 and 3x4 inputs, spectators and clean work use both binding
paths. All 2x3 measurement-history pairs are exhausted.

The unchanged flat scaling emitter retains its narrower real_width>=integer_width
admission. Exact native-flat primitive multiset comparison applies throughout
that common domain. The added negative-fraction range instead uses the unchanged
original CAdd child streams and independent D4 quantization oracle. Original
root models remain unchanged, including their negative counts in extrapolated
formats. For 2x4, the original model reports negative two Toffoli while the
native retained construction emits one And copy. This boundary has explicit
separate resource accounts. Source root decomposition remains unavailable.

The negative-fraction closure uses `g2-truncated-products-wide-` receipts. Final
concentrated regression passes 3,976 checks with one existing IPython warning
and zero skips, plus 17 local integration and 59 focused checks. Strict typing
covers all four batch Python files. Black, Ruff, scope, architecture, IR and
public API checks pass. Five frozen source blobs and the required Qualtran and
Kickmix versions/Simulator are verified. Dependency policy remains at 687
whole-repository findings and zero touched-path findings. The eight accepted
flat/CAdd/Add owners and their parity tests are byte-identical to c06b8aaea.
Default docstring failure evidence is reused after ten-file baseline validation.

## Stored PlusEqualProduct live accumulation, November 20, 2026

`arithmetic/product_accumulation_component.py` closes the occupied-accumulator
stored execution dependency. It preserves unsigned THRU a, b and result with
result_width>=left_width+right_width, and adds or subtracts the full product
modulo the result width. The accumulator can start occupied. The column network
computes its live product/carry parity word in one owned work block, passes that
word directly to the accepted Add or exact Add inverse, then erases the network
and releases its zeroed work. The word can have one absent leading zero column.

`bits/column_steps_component.py` owns the shared typed CX, And compute and
immediate-feedback erasure steps. It serves the existing XOR consumer and this
live Add consumer. Allocation, term generation and consumer choice remain with
their respective owners. The former XOR loop and primitive functions were
extracted without changing their definitions or sequence. Equality checks load
the prebatch weighted-column implementation and compare full Program records
and complete bound records for twelve square widths. The actual resulting
instructions, aliases, allocation/release positions and classical slots match.

Stored and expanded accumulation calls retain exactly one shared Add root,
without a copied intermediate product. Six equal/unequal input and accumulator
formats compare primitive multisets, per-wire order, operands and validated
measurement feedback against the accepted flat accumulation owner. Its Add
child is additionally compared directly with the unchanged original Add or
Adjoint(Add) on the actual live column addresses. Existing original And and
column leaf/channel evidence is reused. Independent global gates can schedule
in different orders, while fresh feedback slots are normalized after validating
each measurement dependency.

Complete coherent input and occupied accumulator values, an entangled spectator
and clean work are checked at widths 1x1 into 2, 1x2 into 3 and 2x2 into 4,
including overflow wraparound. Both directions and both stored/expanded paths
pass. The 1x1 case exhausts all sixteen forward/inverse measurement history
pairs. Other sizes use two trajectory seeds and accepted leaf/channel evidence.
Normal circuit.append(op) round trips pass with the flat root replaced by a
failing stub. Seven accepted flat/shared owners and their existing source-value,
tensor, primitive and classical tests are byte-identical to base fbea65709.

The original PlusEqualProduct supplies value/tensor callbacks and an approximate
T model, with no root decomposition. Its full existing original tensor/value
and basis evidence is reused. Actual gates and approximate model are separately
recorded in `g2-live-accumulation-evidence.json`. At 2x2 into four bits, the
construction emits 36 T/Tdg, 73 CX and nine measurements/resets with nine clean
work bits, versus the original approximate 32 T model. The inverse has the same
physical counts. The live column construction retains the accepted flat counts.

InvertRealNumber's mixed THRU output and generic foreign construction, drawing,
rebinding and serialization remain the complete multiplication-file gaps.
Original root decomposition absence stays explicitly recorded. Q-0208 and
Q-0209 remain open until the remaining full-file scope is disposed.

Live-accumulation receipts use `ftqc-reference-env/g2-live-accumulation-`.
Concentrated regression passes 4,026 checks with one existing IPython warning
and zero skips. Local integration passes 17 checks. Final focused checks pass
47 cases. Strict typing covers four Python files, including the existing XOR
consumer. Black, Ruff, scope, architecture, IR and public API checks pass.
Five frozen source blobs, Qualtran 0.7.0 and Kickmix 0.1 Simulator are verified.
The twelve prebatch full Program and binding comparisons pass exactly.
Dependency policy reports 687 existing whole-repository findings and zero
batch-path findings. Known default docstring failure evidence is reused after
verifying its ten files against fbea65709. Default-suite completion remains open.
Type annotation and unused-import fixes after broad collection are checked by
the final strict and focused runs.

## Stored reciprocal direct dependency, November 20, 2026

The final native numerical root in the eight-class multiplication file now has
stored composition. `reciprocal_seed_component.py` creates a RIGHT pure-fraction
seed or consumes the matching LEFT image. Its Figure 4 selection and actual
stored MultiControlX cleanup retain the existing seed construction.
`reciprocal_newton_component.py` calls the accepted integer square, subtracting
live product accumulator, retained XOR slice, restoring product accumulator
and square inverse. It preserves both mixed source and pure-fraction estimate.
Numerator and square storage are owned across their actual use and restored
before release. Precision may be smaller than the source integer width for a
standalone Newton step, retaining the flat step's modular behavior.

`reciprocal_component.py` preserves the source's two mixed QFxp(n,f) THRU ports,
including occupied result storage. It allocates a seed at precision n, performs
ceil(log2(n)) widened steps, XORs the leading f estimate bits into the mixed
result's low f positions, then erases all history and seed storage. The integer
prefix of the occupied result survives. Below one, its mask is zero. The finite
lower estimate remains strictly below one at a=1. Nonpositive fractional
widths consequently have a zero quantized mask and the typed root is identity.
These are explicit native approximation choices inherited from the accepted
paper construction. The source root supplies no numeric callback or executable
decomposition. Its declared square/product children use different precision
assumptions, as already documented. Exact root gate parity remains unavailable.

Stored and expanded seed streams match the accepted flat Figure 4 construction
in primitive multiplicity, per-wire order, operands and validated feedback.
The original multi-control child evidence is reused. Expansion can assign the
private flag and predicate work in a different order, so comparisons use their
actual physical allocation. Five standalone Newton widths match the flat
primitive multisets. Square, PlusEqualProduct, Add, And and MultiControlX source
child evidence is reused unchanged. Scratch addresses may be reused for other
roles at later lifetimes, so a fixed whole-circuit scratch renaming is
inapplicable to Newton's complete stored schedule.

All raw coherent source/estimate/occupied-result combinations and a spectator
are checked for the five small Newton formats, including modular behavior
outside the numerical approximation domain. Seed tests cover creation and LEFT
erasure. Complete mixed reciprocal inputs and occupied results are checked at
(2,1), (3,1) and (3,2), plus nonpositive fraction identities. Whole-state
comparison includes all restored work. Every smallest Newton measurement
history is enumerated, totaling 256. Larger small domains use complete measured
And channel certificates for all basis inputs and two primitive trajectory
seeds, which also check complex relative phase. Root circuit.append(op) round
trips execute through the sparse test interpreter, and the smallest Newton
round trip uses normal dynamic trajectory execution. Flat root functions are
replaced by failing stubs during replacement acceptance.

| Mixed reciprocal n,f | Observed primitives and storage | Original analytical model |
| --- | --- | --- |
| 2,1 | 304 T/Tdg, 699 CX, 76 measurements/resets, 31 peak clean bits | Original graph fails through its minimum-three-bit square child |
| 3,1 | 1,416 T/Tdg, 3,449 CX, 2 CCX, 354 measurements/resets, 62 peak clean bits | 21 Toffoli, 5 And, 46 Clifford, 5 measurements |
| 4,2 | 2,536 T/Tdg, 6,374 CX, 2 CCX, 634 measurements/resets, 99 peak clean bits | 47 Toffoli, 7 And, 66 Clifford, 7 measurements |

Observed work lifetimes balance exactly. Seven frozen source blobs, Qualtran
0.7.0 and Kickmix 0.1 Simulator presence are verified. Eleven accepted owners
and tests are byte-identical to 00b000446. Dependency policy has 687 existing
whole-repository findings and zero batch-path findings. Concentrated regression
passes 4,112 checks with one existing IPython warning and no skips. Local
integration passes 17 checks. Final reciprocal-focused regression passes 86
checks. Strict typing covers all four batch Python files. Black, Ruff, scope,
architecture, IR and public API checks pass. Default docstring failure evidence
is reused after verifying its ten implicated files against 00b000446. Full
default-suite completion remains open. Receipts use
`ftqc-reference-env/g2-reciprocal-components-`.

All eight native typed arithmetic execution owners are now delivered. Q-0208
and Q-0209 remain open for the complete foreign-interface disposition of
construction, drawing, rebinding, serialization and original missing root
decomposition/autotester modes. The whole-file inventory must resolve those
remaining interfaces before parent acceptance.

## Complete multiplication-file interface closure, November 20, 2026

The complete frozen module has eight classes, forty own methods and eight
factories, totaling 56 definitions. Its complete native disposition follows.

| Own definitions | Native owner and final disposition |
| --- | --- |
| PlusEqualProduct validation, signature, a/b/result dtype properties, registers, adjoint, apply, classical callback, tensor, wire symbol, Cirq diagram info, graph | The stored live accumulator, existing classical/tensor owners and analytical T model retain these contracts. The SDK adapter converts both adjoint flags into the same native Program. Diagram results use the original Cirq label c+=a*b or c-=a*b. |
| PlusEqualProduct.with_registers | The unchanged original explicitly raises NotImplementedError. Native physical rebinding uses the existing named emitter with reversed wires and fresh feedback slots. This source refusal is fully disposed. |
| Square signature, classical callback, wire symbol, graph, tensor, adjoint | The accepted stored square and value/tensor owners supply the native contract. Its previous SDK square realization is byte-identical and remains the same owner. |
| SumOfSquares signature, wire symbol, graph | The shaped stored sum owner, retained model and row-major score geometry cover all three methods. |
| Product signature, wire symbol, graph | The stored product owner, retained analytical model and a*b score label cover all three methods. |
| ScaleIntByReal signature, wire symbol, graph | The stored D4 owner retains mixed result metadata, including the accepted wider integer case. The SDK adapter uses that owner. Source formula and actual primitive counts remain separate. |
| MultiplyTwoReals signature, wire symbol, graph | The stored D5 owner, retained analytical model and pure-fraction metadata cover all three methods. |
| SquareRealNumber validation, signature, wire symbol, graph | The stored D24/D28 owner preserves both THRU input ports and RIGHT/LEFT result ownership. Minimum width and analytical model are retained separately from the executable construction. |
| InvertRealNumber validation, signature, wire symbol, graph | The mixed THRU reciprocal, owned seed/Newton history and retained eight-child source model provide the final native owners. Approximation and source-model precision differences remain explicit. |
| Eight factories and module documentation specifications | The accepted native catalogue, full notebook and source declaration tests cover all factories, signatures, fixed-point formats and call graphs. Original source factory serialization passes for all eight. |

`ecosystem/cirq/_multiplication_arithmetic.py` translates the seven remaining
concrete source declarations into their existing Compiler Program owners.
The SDK receives primitive native IR immediately. Source objects stop at this
boundary. The bridge uses the existing Program expansion and physical binder,
with a synthetic pool bounded by the sum of allocation lifetimes. Only scratch
wires actually used by binding are allocated in the SDK. RIGHT results receive
local clean storage. LEFT images transfer ownership for erasure. Actual output
aliases are restored before returning to the original SDK operands. This
avoids maintaining duplicate arithmetic or resource formulas in Ecosystem.
The existing integer-square bridge remains byte-identical.

`drawer/arithmetic_symbols.py` owns multiplication formulas and register labels.
Existing layout handles array grouping, lifetime arrows and geometry. All eight
complete scores match original geometry, ordered ports and formula labels,
including SumOfSquares' shaped input. Native snapshots serialize and deserialize
through the existing CircuitIR schema. Every original factory also passes the
unchanged source serialization check. The deliberately unsupported original
with_registers callback is preserved as a source limitation, with native named
physical rebinding separately verified.

Forward and backward imported snapshots cover all eight roots. Normal append,
reversed logical/work placement and fresh classical slots preserve complete
complex input states and their spectator through round trips. Forward states
also match independent integer or accepted truncation/reciprocal oracles.
Complete all-input measured-channel certificates verify each erase invariant
and zero scratch. Original accumulation value/tensor and square tensor
callbacks retain their prior direct evidence. Observed primitives remain the
native paper streams. The original roots still have no executable decomposition.
No original-root primitive alignment is asserted for those missing recipes.
Physical capture retains its existing concrete positive-port contract. Symbolic
metadata and resource formulas are covered by the earlier declaration owners.
Generic inherited control protocols and arbitrary foreign subclasses belong to
their own source-file owners.

| Imported example | Actual forward primitives and clean/slot requirements | Original model |
| --- | --- | --- |
| PlusEqualProduct 1x1 into 2 | 8 T/Tdg, 14 CX, 2 measurements/resets, 2 clean and 2 slots | Approximate 8 T |
| Product 1x1 | 4 T/Tdg, 7 CX, 1 measurement/reset, 1 clean and 1 slot | 1 Toffoli |
| SumOfSquares two one-bit words | 4 T/Tdg, 10 CX, 1 measurement/reset, 1 clean and 1 slot | 1 Toffoli |
| ScaleIntByReal 2,1 | 4 T/Tdg, 6 CX, 0 measurements, 0 clean, 1 reusable inverse slot | 1 Toffoli |
| MultiplyTwoReals 2 | 4 T/Tdg, 6 CX, 0 measurements, 0 clean, 1 reusable inverse slot | 1 Toffoli |
| SquareRealNumber 3 | 8 T/Tdg, 16 CX, 2 measurements/resets, 2 clean and 2 slots | 0 Toffoli in the original formula |
| InvertRealNumber 2,1 | 304 T/Tdg, 699 CX, 76 measurements/resets, 31 clean and 76 slots | Original minimum-three-bit square failure |

The original test file passes all 24 runnable non-notebook checks. Its 24 skips
are exactly the original missing-decomposition, decomposition-count and dtype
modes. The accepted complete notebook run is reused. Native concentrated
regression passes 4,234 checks with one existing IPython warning and no skips.
Local integration passes 17 checks. Final focused checks pass 30 cases. The
additional Cirq 1.6.1 lane passes 21 checks, alongside the default Cirq 1.7 lane.
Strict typing covers all five batch Python files. Black, Ruff, scope,
architecture, IR and public API checks pass. Six frozen source blobs and the
required Qualtran/Kickmix Simulator environment are verified. Eleven accepted
owners/tests remain byte-identical to 7fc0bef3d. Dependency policy reports 687
existing whole-repository findings and zero batch-path findings. The interrupted
over-broad run is retained separately and excluded from acceptance. Default
coverage continues using the verified pre-existing docstring failure baseline.
Receipts use `ftqc-reference-env/g2-multiplication-foreign-`.

Q-0208 and Q-0209 have complete definition, dependency, test and asset
disposition and can be accepted together with the previously accepted Q-0207
notebook. Native numerical constructions and the upstream model-only boundary
remain independently traceable.
