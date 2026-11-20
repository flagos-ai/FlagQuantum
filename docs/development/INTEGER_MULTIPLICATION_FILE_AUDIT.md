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
