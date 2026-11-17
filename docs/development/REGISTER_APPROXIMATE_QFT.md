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
