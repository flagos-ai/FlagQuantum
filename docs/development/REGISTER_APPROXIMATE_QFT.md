# Precision-limited phase-gradient QFT

The emitter follows the [locked ApproximateQFT](https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/qft/approximate_qft.py)
loop. At data position i it retains the min(i, precision - 1) nearest
earlier qubits, reverses them for the big-endian adder, and uses the
current qubit as the control. The add targets a gradient prefix and is
shifted right once. A Hadamard follows each addition. The optional final
swaps reverse the result order. The adjoint reverses that order and
subtracts each addend. An empty retained window needs no addition.

Data and gradient are caller-owned. The negative gradient eigenstate
gives the truncated Fourier transform and returns to its initial state.
The shared controlled adder restores the borrowed work. A one-qubit
data register uses only a Hadamard and can take an empty gradient.
The concrete epsilon formula follows the locked source.

With five data qubits and three gradient qubits, the emitted arithmetic
uses 56 T-family gates, 14 measurements, four clean work qubits and
four reusable slots. The source estimates 24 T for the same loop.

## Whole-file evidence

| Ledger | Locked file | Git blob | Coverage |
| --- | --- | --- | --- |
| Q-0636 | `approximate_qft.ipynb` | `4481632e0416599a9f2841e2b8bd26bc477270b2` | Both source examples, graphical signatures, concrete call graph and symbolic T-cost expression |
| Q-0637 | `approximate_qft.py` | `e6dfcee8614cf80463c97adfbe282daa239c3dc4` | Existing executable cutoff, precise shifted-add operands, default and epsilon-derived precision, and raw concrete and symbolic costs |
| Q-0638 | `approximate_qft_test.py` | `7dd0b10f8da35fc476e0d49dda2158c8ada93535` | Original tests plus the native model and execution evidence below |

The native declaration drops a zero-bit gradient port, as the locked
signature does for the one-wire default. Concrete gradient precision
reuses `approximate_qft_precision`; the symbolic expression preserves
`ceil(log2(n*pi*sqrt(2)/epsilon))`. Direct call keys retain the source
window and gradient widths, with shift one and positive control. The
concrete graph combines equal window sizes; the symbolic graph retains
the source's grouped `n`-fold phase add. The notebook's symbolic T count
uses four modeled T gates per declared Toffoli.

All seven original notebook code cells executed. The native notebook
executes eight code cells, renders two SVGs without upstream imports,
and checks a three-wire truncated transform through `Circuit.append`.
Its example ports use the source widths `(6, 5)` and `(50, 25)` at
epsilon `1e-5`. The original tests passed 30 cases, including the
seven-wire approximation bound in the reference simulator. The native
tests exercised exact gate order and original operand binding, complete
small complex branch and gradient restoration, both directions, source
precision checks to width 10,000, and cost and width checks to 123. All 30 focused cases and 5,520
concentrated regressions passed. The expected deprecation warning comes
from the original symbolic T-count helper. Strict typing, format, scope,
architecture, internal IR and public API checks passed.
They did not run the full seven-wire approximation case on the native
dynamic statevector backend.

At width five with gradient precision one, the source declares four
zero-width phase-add calls and reports minus eight Toffoli gates. The
raw source model preserves this diagnostic value. The actual native
stream is five Hadamards and two swaps with no phase-add operations,
measurements or borrowed work. Q-0637 stays open for the source cost
boundary. Q-0638 stays open because two upstream automatic count checks
are disabled and reported as expected failures.

| Data and gradient widths | Source model Toffoli | Source modeled T at four per Toffoli |
| --- | ---: | ---: |
| 5 and 3 | 6 | 24 |
| 6 and 5 | 18 | 72 |
| 50 and 25 | 1,702 | 6,808 |

These estimates do not measure emitted gates. The same five-data-wire,
three-gradient-wire native case emits 56 T-family gates and 14 measurements.
Its AddIntoPhaseGrad leaves have executable arithmetic and a resource
model, with no gate-level decomposition. Tests compare the original
window and operand bindings, the full small source joint operator,
raw measurement histories, and the truncated kickback with a prepared
and subsequently cleared gradient.


## Stored catalog dependency interface

This batch retains the complete locked source-file graph through the existing
Program and append paths. Words use MSB-first physical labels and caller-owned
through storage. Textbook q is unsigned. Gradient QFT and approximate QFT keep
a separate unsigned fractional gradient through word owned by the caller.
Shared multiplication and controlled addition own their measured carry work,
which must return to zero. TwoBitFFFT has x/y QBit through ports and no work.
Its original ignored-adjoint decomposition remains separate from the native
exact inverse. Existing primitive execution and source-model evidence is reused.
Compiler owns stored word/call composition, Algorithms owns the already accepted
fermionic Fourier sequence, and the example provider owns catalog glyphs and
literal-definition qualification. No protected contract change is required.


## Complete catalog graph closure

Both complete source factory roots retain all 32 and 270 ordered calls.
Their controlled shift-one AddIntoPhaseGrad children remain original atomic
refusals. Shared native CAdd and its exact adjoint own measured work below
those boundaries. Final swaps disjoint from the last H occupy its Cirq EARLIEST
moment; the swap touching that last data bit follows. The stored graph retains
this source order while the existing flat emitter remains an independent
execution path. Six-bit/five-gradient-bit literal execution uses 112 T/T-dagger,
232 CX and 28 measurements on nineteen physical wires. Fifty-bit/25-gradient-bit
execution uses 7200 T/T-dagger, 16004 CX and 1800 measurements on 123 wires.
Separate original models are eighteen/1702 Toffoli, T-equivalent 72/6808 and
QubitCount eleven/75. Complete small joint operators and every forward and
existing-emitter inverse raw history pass. The large factory uses exact stored
calls and inventories. Stored construction also covers one-bit gradients and the successful one-data-bit,
zero-gradient case. The original negative-model boundaries remain separately
recorded under Q-0637.

An independent immutable-source probe confirms successful stored decompositions
for `(width, precision)` equal to `(1, 0)`, `(1, 1)`, `(2, 1)` and `(3, 1)`.
The first contains one Hadamard and has no gradient port. The one-bit-gradient
cases retain the caller gradient, with zero-width phase-add input omitted from
the child signature. `(3, 0)` raises the original `KeyError('phase_grad')`.
The stored native constructor now covers every successful case in this probe.
The empty input of each controlled phase-add call omits its x port and emits
no primitive, measurement or workspace allocation. Native multi-bit zero
precision retains an explicit ValueError at construction.


## Concentrated catalog acceptance

The four-file Fourier group retains thirteen exact source-success recursive
call graphs and original drawings across five concrete roots. Immutable source
atomic refusals are retained explicitly. Complete joint amplitudes, applicable
forward and inverse raw histories, noncontiguous bindings, an entangled
spectator, clean work, normal append, Program copy and IR JSON pass in the
1095-check execution regression. Relevant local integration passes 29 checks.
Strict typing covers eleven source files, with formatting, scope, architecture,
IR and API checks passing.

Catalog acceptance covers 881 checks across the retained first 47 complete
Catalog/Chemistry checks and 834 resumed checks. The initial full run reached
its twenty-minute limit after those two complete files passed. The terminated
process was verified absent before resuming from Rotations. Three existing
deselections and fourteen notebook warnings remain recorded. Execution has two
original deprecated-cost warnings. The separate resource receipt keeps literal
primitive inventories, used clean storage and original analytical models apart.


## Small-precision stored graph closure

Ten concrete cases cover widths one through four, zero or one gradient bit and
both reversal settings. Ordered source calls, operands, exact source drawings,
complete complex joint operators, an entangled spectator, forward and exact
inverse execution, append, Program copy and lowered JSON pass. These channels
have one empty measurement history and no work. Width one uses one H. Wider
one-bit-gradient cases use one H per data bit and, when requested, one SWAP per
reversed pair. Actual physical width is data width plus gradient width.

The original model retains negative Toffoli estimates at one gradient bit,
including -2, -4 and -6 at data widths two, three and four. Its separate width
estimates are four, five and six, while literal execution needs three, four and
five wires. No negative estimate is interpreted as a physical instruction count.

Concentrated affected regression passes 130 checks, fresh complete Fourier
source/drawing/complex-state checks pass 32, and local integration passes 29.
Exact ProgramRecord comparisons with published d3ec41d32 preserve 256 positive
phase-add Programs and 24 pre-existing approximate-QFT Programs, including all
stored definitions. The preceding 881-check complete catalog evidence is reused
for the unchanged registered factories. Earlier strict diagnostics for test
tuple annotation and optional drawing narrowing were corrected before acceptance.
