# Rz via phase gradient source-file audit

The reference is unchanged Qualtran v0.7.0 at commit
`bcb17bb25360ed68ce3bc084645db3e929799389`.

| File | Frozen Git blob | Ledger item |
| --- | --- | --- |
| `rotations/__init__.py` | `e6c63bdf43262fd6e3771a9eebd5c8e7ad75d4b8` | Q-0666 |
| `rz_via_phase_gradient.ipynb` | `75e37a757fb74d78b06704638dd34a78cb42c7c2` | Q-0684 |
| `rz_via_phase_gradient.py` | `d3ec380d24af7dfb8a115b6fe88074ad53e22e93` | Q-0685 |
| `rz_via_phase_gradient_test.py` | `4fb5e79871ad3943caad86206771575039146f7d` | Q-0686 |

Run `jupyter notebook examples/notebooks/rz_via_phase_gradient.ipynb`.
The native notebook preserves the original ten cells and appends the
four-bit operation through `circuit.append(op)`. It renders the
signature and source-declared call graph and prints both the literal
gate inventory and the symbolic resource formula. Both notebooks ran in
independent kernels, and the native kernel rendered two SVGs with
Qualtran, Cirq and quimb imports forbidden.

## Source binding and tensor evidence

The qubit, angle and phase gradient are through registers. The bits of
both fixed-point words are cast through unsigned integers at the same
width. The default branch subtracts on zero and adds on one. The adjoint
flips those roles. The source graph performs two casts on entry,
one ControlledAddOrSubtract call, and two casts on return. The tests
compare the original operations, physical operands, and feedback slots
at the same arithmetic lowering depth.

For a raw angle word `a` and a negative-exponent Fourier gradient of
width `b`, the body applies `Rz(4πa/2**b)` to the data qubit.
The caller supplies the angle and Fourier gradient. Carries begin at
zero and return to zero in every measurement branch. The original
three- and four-bit tensor cases prepare angle words 1 and 2. The
native tests compare two independent complex input vectors to the full
source tensor and restore all initialized registers on every raw
history. The four-bit body also covers every angle word, gradient basis
word, and control value in both directions.

The unchanged source suite returned eight passes and one expected
failure from the source counts autotester. The original source symbolic
cost assertions were checked separately against the shared Add model.

| Four-bit body | Literal T/T† | Literal CX | Literal measurements | Source QEC categories | Reserved / source wires |
| --- | ---: | ---: | ---: | --- | --- |
| Forward | 12 | 41 | 3 | 3 And, 28 Clifford, 3 measurements | 12 / 12 |
| Adjoint | 12 | 41 | 3 | 3 And, 26 Clifford, 3 measurements | 12 / 12 |

For a symbolic equal width `n`, the source forward model uses `n-1` And,
`9n-8` Clifford, and `n-1` measurements. The adjoint removes the two
polarity flips. The symbolic live width follows the source maximum over
the input, the Add child and each controlled complement. One-bit
Clifford counts preserve the original abstract Add anomaly.

## Remaining source boundaries

Q-0684 and Q-0686 are covered. Generic dtype bit binding and explicit
resolution of symbolic widths are now exercised below. The source Add
validator raises for incomparable symbolic widths, such as angle width
one with an unconstrained symbolic gradient width. The resource query
preserves that admission boundary. Q-0666's export inventory is accepted.

## Whole-file generic object binding

The entire source class, factory, documentation specification and original test
file have been read. The two dtype fields feed the source signature and its two
unsigned-cast properties. Native `RegisterCast` preserves those same raw words;
`rz_via_phase_gradient_ir` delegates the arithmetic to the existing measured
controlled-add/subtract owner. The source body has four casts and one arithmetic
child, while its declared call graph omits the casts. The native example and
independent symbolic model preserve that distinction. The factory, original
tensor helper, symbolic cost assertion and notebook map to the previously
accepted tutorial and file tests.

`test_rz_source_binding.py` creates the original class with symbolic width fields,
rejects unresolved physical execution, and explicitly resolves its dtypes. Actual
`QUInt`, `QAny`, mixed fractional `QFxp`, signed `QInt`/`QFxp`, and bounded
`BQUInt` objects bind in both directions at widths (1,1), (2,2) and (2,3).
The original casts determine the unsigned bit interpretation. Signed numerical
values are not reinterpreted as signed arithmetic. Bounded padding columns
exercise the physical circuit extension; they do not enlarge the bounded
register's declared value domain. A one-bit signed `QFxp` is invalid in the
source constructor, so that case uses the valid one-bit `QInt` type.

Imported and native emitters use the same `BlackBoxPrepare` append consumer.
Owned JSON IR, every logical matrix column, all-history channel identities,
independent final-zero checks, exact opcode histograms and true adjoint recovery
agree. Every raw angle and both data values also satisfy the analytic complex
phase when applied to the negative-exponent Fourier gradient. Each direction's
independent resource model agrees with the unchanged source. Existing operand,
dependency and measurement-feedback evidence is reused at the accepted lowering
depth. Angle width greater than gradient width remains rejected by the original
arithmetic admission rule.

| Actual source widths (angle, gradient) | Forward / adjoint instructions | T/T† | CX | Measurements | Physical wires |
| --- | ---: | ---: | ---: | ---: | ---: |
| (1,1) | 5 / 3 | 0 | 3 | 0 | 3 |
| (2,2) | 26 / 24 | 4 | 13 | 1 | 6 |
| (2,3) | 50 / 48 | 8 | 26 | 2 | 8 |

The one-bit abstract Add Clifford anomaly is retained in the source model;
the physical table is measured separately. Q-0685 therefore retains this
inherited model boundary and the incomparable symbolic-width admission case.
The new generic and resolved-symbol binding evidence does not claim execution
of an unresolved symbolic circuit or close the parent while those boundaries
remain in its ledger.

The final regression passes 656 cases in 39.25 seconds. Strict typing, Black,
Ruff, Algorithms scope and diff checks pass. This tests/audit-only batch reuses
accepted architecture, IR, protected API and independent kernel evidence.
An initial fixture tried the source-invalid signed one-bit `QFxp`; replacing
that fixture with the valid `QInt(1)` preserves the source admission rule.

## Complete stored catalog dependency group

The native catalog now retains the complete original five-call root in a typed
Program. Compiler owns execution composition. The example provider owns labels,
source-title fallbacks and presentation qualification. Existing Cast, controlled
word XOR, conditional Add/Sub and phase-addition owners remain authoritative.
Users append the resulting bound operation through `circuit.append(op)`.
Words are MSB first. Input words remain caller owned. Angle allocation and
measured carries are owned and released by their respective components.

The recursive observer verifies complete ordered calls, register descriptors,
operands and qualified child definitions against unchanged source. Mutating CX
operands rejects source labels. Native helpers below original atomic arithmetic
refusals are inventoried separately. Shared Program scheduling can reorder
disjoint primitives and allocates classical slots in measurement order. Exact
primitive comparison maps each measured wire to that slot before checking every
wire's gate order and feedback.

Both concrete operations and their exact inverses were tested on the full
complex logical tensors at widths two and three, with reversed/noncontiguous
bindings and an entangled spectator. Every forward and inverse raw measurement
history preserves the complete complex amplitudes and returns all work to zero.
The normal DynamicCircuit append path, Program ownership copy and lowered
CircuitIR JSON round trip pass. Existing larger-width, source-binding and SDK
execution evidence is reused in the concentrated regression.

Symbolic precision retains its logical five-call graph and bodyless typed
children at original decomposition refusals. Symbolic physical binding and
unresolved symbolic angles retain their recorded refusal boundaries. The source
reference is unchanged. This catalog closure leaves the parent source boundary
items open.

The batch receipts under `ftqc-reference-env` use the prefix
`g2-phase-rotations-ui-` and date `20261122`. The inventory records eight exact
recursive graphs, 25 original atomic contexts and 75 native-only helpers across
three source-success factory contexts. Three source drawing failures arise from
the original controlled BitwiseNot/XorK title type. Fresh comparable drawings
remain exact. Native fallback drawings are explicit.

The Rz factory contributes five exact graphs and 15 atomic contexts. Its root
and three comparable descendant drawings match. The conditional arithmetic
child retains the original BitwiseNot title failure. The actual four-bit body
uses 2 X, 41 CX, 9 H, 6 T, 6 T-dagger, 3 S, 3 measurements, 3 conditional CZ and
3 resets, with three clean carry allocations/releases and peak width 12.
The original model separately declares 3 And, 28 Clifford, 3 measurements and
width 12. Its default T-equivalent estimate is 12.

Batch acceptance returned 838 concentrated catalog passes, three existing
deselections and 14 existing notebook syntax warnings; 326 execution/SDK passes;
and 29 relevant local integration passes. Strict typing of seven changed Python
files, Black, Ruff, ownership scope, architecture, IR and API checks pass.
Fourteen frozen source assets match their inventory blobs. The first catalog
run exhausted disk during reproducible export; its diagnostic is retained and
the successful run followed HTTP-cache and partial-export cleanup. An additional
broad integration probe passed 163 cases before local Gloo address resolution
failed. That environment diagnostic remains separate from the passing relevant
integration suite.


## Complete faithful source-file closure — 2026-11-28

Q-0685 is accepted under the approved G2 faithful-migration rule after closing
the source-success symbolic-interpretation Cast gap. Earlier open status based
solely on source model anomalies or source-refused physical dimensions is
historical. No generic transport, SDK or workflow parent closes here.

| Complete source responsibility | Native owner and evidence |
| --- | --- |
| Both dtype fields and ordered THRU signature | Existing format constructor and typed `q`, `angle`, `phase_grad` ports preserve nominal descriptors and MSB order. |
| Both unsigned dtype properties | Register sizes feed existing unsigned descriptors; arithmetic uses raw bits. |
| Composite body | Four owned Cast views surround the existing ControlledAddOrSubtract call, preserving source producer/operand/result edges. |
| Declared call graph | Existing resource model retains the single arithmetic child and original omission of Cast costs. |
| Example and documentation specification | Accepted four-bit tutorial, original tests, drawings, notebook and source-free append example are reused. |

The actual original QFxp(3,f) root and full 128-by-128 tensor succeed. The native
composer previously treated symbolic fractional metadata as an unresolved wire
size and supplied two bodyless Casts. It now checks physical dimensions and
uses the existing raw Cast owner when those dimensions are known. Approved
Proposal 130 permits this view. The original descriptor remains attached to
every port; no numerical fixed-point interpretation or symbolic arithmetic is
added. Casts add no gates, measurements or workspace. Unresolved wire dimensions
remain declarations and fail physical binding.

The 24 profiles cover widths two/three, angle-only, gradient-only and both-word
metadata, signed/unsigned fractional expressions, bounded iteration lengths and
Montgomery moduli. Source five-call roots, complete complex operators, every
small forward and inverse history, coherent spectators, zero carry work and
nominal output aliases agree. Primitive instructions and lifetime events equal
the accepted unsigned raw-storage implementation, whose unchanged original
primitive/operand/feedback evidence is reused. Actual source constructor wires
at depths zero/one/two also bind identically for the representable profiles.
The original Montgomery writer drops its modulus; no faithful object roundtrip
is claimed for that codec. Native construction and append work with source SDK
imports blocked.

Independent intake additionally checks complete original raw operators and exact
accepted native instructions/events for GF(3), GF(5), GF(9) and a packed GF(3)
polynomial. This is unsigned raw-bit reinterpretation under the Rz source
contract, not finite-field addition or external value admission. Padding codes
are physical storage columns, not new valid bounded/field numerical values.

Original incomparable-width admission, empty omitted-port failures, symbolic
physical extents, atomic arithmetic boundaries and one-bit Add model anomalies
remain explicit. No original graph, callback, tensor or test was patched.
Measured inventories remain separate from source QEC categories and peak widths.
Existing signed, unequal-width, eight-bit, full small-history and original
asset evidence is reused. Shared receipts use
`g2-phase-rotations-full-closure-*20261128`; metadata and field/literal intake
receipts retain the exact supported/refused profiles.

Batch acceptance passes 134 concentrated source/execution cases, 47 independent
Cirq 1.6.1 cases and 32 relevant local integration cases. Strict typing of two
Python files, Black, Ruff, Compiler scope, architecture, multi-level IR, required
checks, public API baseline, English, links and whitespace pass. Twenty-five
complete immutable source/asset/dependency blobs and real Qualtran 0.7.0/Kickmix
0.1 Simulator are verified. Twenty-seven resource rows separate literal
primitive inventories and source models. Ten actual optional/required dependency
absence scenarios pass. Global dependency findings remain at 570 in 362 files,
with none in the new consumer; Q-0009/Q-0011 remain open. No source patch,
checker waiver, new exclusion or CI-wait dependency was introduced.
