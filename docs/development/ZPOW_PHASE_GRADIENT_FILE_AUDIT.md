# ZPow via phase gradient source-file audit

The reference is unchanged Qualtran v0.7.0 at commit
`bcb17bb25360ed68ce3bc084645db3e929799389`.

| File | Frozen Git blob | Ledger item |
| --- | --- | --- |
| `rotations/__init__.py` | `e6c63bdf43262fd6e3771a9eebd5c8e7ad75d4b8` | Q-0666 |
| `zpow_via_phase_gradient.ipynb` | `52d26d103692d41efd47ca719ad24102cf984b79` | Q-0687 |
| `zpow_via_phase_gradient.py` | `b9fe0ca61754f68aef838ef23abf2e6ab14e7690` | Q-0688 |
| `zpow_via_phase_gradient_test.py` | `17fa1e59be1b4d5b0b845531897485f688f41724` | Q-0689 |

Run `jupyter notebook examples/notebooks/zpow_via_phase_gradient.ipynb`.
The native notebook retains the twelve original cells, displays the
three original signatures and declared calls, and appends the 40-bit
body on caller-owned wires. Both notebooks ran in independent kernels.
The native kernel rendered two SVGs with Qualtran, Cirq and quimb imports
forbidden.

## Source binding

The original precision `1e-11` selects 40 gradient bits. The data and
gradient registers are through. The quantized half-exponent for the
original `3/8` angle is `206158430208`. The source allocates a clean
angle word, applies the first controlled XorK, adds the loaded value to
the gradient, applies the same XorK again, and frees the cleared word.
The native compiler borrows one angle word and the measured adder
carries. The adjoint reuses the same constant and reverses the direction
of the phase addition. The tests compare the bound source child order,
operands and classical feedback slots in both directions.

The 40-bit body was certified on both data values and four representative
gradient words including wrap. Sparse complex trajectories check the
full supplied state with all clean storage returned. The original three-,
four- and seven-bit tensor cases were reproduced with exact relative
phase from the stored constant. The three- and four-bit cases project
every measurement history. The seven-bit certificate covers every
gradient basis word and both data values, followed by two sparse
coherent executions on each input vector. The ideal rotation comparison
uses the original suite's global-phase allowance. The original suite,
including its notebook, passed 19 cases and skipped three checks for
symbolic decomposition, inferred counts or dtype checking.

## Literal resources and model

| 40-bit body or source model | T/T† | CX | Measurements | Declared QEC categories | Wires |
| --- | ---: | ---: | ---: | --- | --- |
| Native measured body | 156 | 469 | 39 | — | 120 reserved |
| Source concrete | — | — | 0 | 38 Toffoli, 4 Clifford | 81 logical |
| Source symbolic angle | — | — | 0 | 38 Toffoli, 80 Clifford | 80 logical |
| Source symbolic precision | — | — | 0 | `b-2` Toffoli, `2b` Clifford | Source live-width maximum |

The symbolic angle uses the source call-graph width fallback. The
symbolic precision reaches the composite width query, which keeps the
maximum over its source children. Both use the unencoded half-exponent
as the controlled-load declaration. Native execution binds concrete
parameters before emitting gates.

## Remaining source boundaries

Q-0687 and Q-0689 are covered. Q-0688 retains unresolved symbolic execution and
the unsigned angle format boundary. Explicit factory resolution is covered below.
A negative forward exponent is
rejected in the source unsigned conversion. Native reversal uses the
adjoint of the nonnegative encoded operation. Q-0666 remains open
while its other rotation exports retain their registered boundaries.

## Complete factory resolution and replacement

The complete class, three example factories, documentation specification and
original test file map to the existing native catalog, compiler and file tests.
The two source symbolic factories are resolved explicitly with `attrs.evolve`
before import. The same operation binds the concrete factory to the small
validation cases. Native execution uses `circuit.append(op)` and the existing
controlled-XOR and measured phase-addition owners. The source reference remains
unchanged.

`test_zpow_source_binding.py` covers widths one, two and three with exponent
`3/8`, and three-bit exponents `1.25` and `3.0`. The latter exercises the original
unsigned wrapping conversion. Each resolved object is imported in both directions,
round-tripped through owned JSON IR and replaced by the native emitter at the same
named append consumer. Complete logical operators, exact primitive histograms,
all-history channel certificates and independently checked final-zero storage
agree. Explicit projections enumerate every raw measurement history on a complex
superposition of all logical words. Every Fourier-gradient input also has the
analytic complex phase from the stored half-exponent. The existing source-graph
walker checks child order, physical operands and feedback at the shared arithmetic
lowering depth. Resource categories and live width are checked independently
against the original model.

| Concrete width and exponent | Literal T/T† | Literal CX | Literal measurements | Reserved wires |
| --- | ---: | ---: | ---: | ---: |
| 1, 3/8 | 0 | 1 | 0 | 3 |
| 2, 3/8 | 4 | 9 | 1 | 6 |
| 3, 3/8 | 8 | 23 | 2 | 9 |

The same histograms hold in both directions. These measured primitive counts
remain separate from the source abstract Toffoli/Clifford model above. The
one-bit source Add model anomaly is inherited. An unresolved symbolic precision
fails physical shape validation, while an unresolved symbolic angle reaches the
source decomposition rejection. Negative forward exponents fail the original
unsigned conversion and the native admission check. These cases remain explicit
boundaries under Q-0688. The previously accepted 40-bit body, original tensor
tests, independent notebook and symbolic model evidence are reused.

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

Concrete ZPow contributes two exact graphs and six atomic contexts. Symbolic
precision contributes one exact graph and four atomic contexts. Both roots
retain their original XorK title failure. The concrete controlled-XOR child
has an exact fresh source drawing. Symbolic-angle decomposition is refused.
The 40-bit factory uses exactly 469 CX, 117 H, 78 T, 78 T-dagger, 39 S,
39 measurements, 39 conditional CZ and 39 resets. Forty angle/carry allocation
and release events retain the caller's data and gradient, with peak width 120.
The source model separately declares 38 Toffoli, 4 Clifford and width 81.
The existing 40-bit sparse coherence certificate is reused alongside the new
stored-Program exact primitive and feedback comparison.

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

Q-0688 is accepted under the approved G2 faithful-migration rule after reviewing
the complete source and all direct dependencies. Earlier open statements based
solely on original refusal/model boundaries are historical. No Z-power emitter
changes in this batch; accepted complete source evidence is reused.

| Complete source responsibility | Native owner and evidence |
| --- | --- |
| Exponent, precision width, signature and gradient dtype | Existing format constructor and fixed-point through ports preserve metadata. |
| `from_precision` | Existing exact ceil/log2 precision helper, numeric and symbolic factory evidence retain the original width rule. |
| `_load_bloq` | Existing unsigned truncating/wrapping encoding for concrete widths; symbolic width retains the original unencoded half-exponent. |
| Composite root | Owned Allocate, controlled load, AddIntoPhaseGrad, repeated unload and Free preserve all five ordered calls. |
| Call graph and display | Existing model retains two controlled loads plus one atomic addition; catalog/provider owns labels and original drawing refusals. |
| Three examples and DocSpec | Accepted concrete, symbolic-precision and symbolic-angle factory paths, original tests and notebook are reused. |

New literal-boundary intake verifies that SymPy Integer(3) takes the source
symbolic-width path: positive and negative numeric exponents produce successful
five-call roots but leave noninteger XorK constants. Original tensor simulation
then refuses that loader. Native roots preserve the exact source producer,
operand and result edges with explicit unresolved children. Four literal and
free-symbol roots are checked; no invalid loader is silently encoded as a
concrete integer. Symbolic exponent decomposition is refused, while zero width
retains the original missing-phase_grad error. A free symbolic width has no
physical size. These source limits do not require an invented numerical body.

Concrete negative exponents retain original unsigned encoding refusal; the
accepted exact native inverse preserves the valid nonnegative load constant and
reverses addition. Original one-bit/negative small-width model anomalies,
Cast omissions and source title errors remain separate. The 40-bit structural
and sparse coherent certificates, exact primitive and feedback comparisons,
small full tensors, all applicable histories and work cleanup remain valid.
Source ideal-rotation comparisons that allow a global phase remain distinct
from exact source-operator comparisons; no absolute-phase claim is inferred
from that looser original assertion. No generic scalar, codec or SDK parent
is waived. Pending Proposal 131 remains unchanged.

Measured 40-bit inventory remains 156 T/T-dagger, 469 CX and 39 measurements
on 120 physical wires. The original independent model remains 38 Toffoli,
four Clifford and peak 81. Shared acceptance/resource/frozen-file receipts use
`g2-phase-rotations-full-closure-*20261128`.

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
