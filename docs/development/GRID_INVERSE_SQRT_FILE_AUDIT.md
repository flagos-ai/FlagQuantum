# Grid inverse-square-root file audit

This batch covers the initializers, inverse-square-root declarations and original tests in the grid Hamiltonian package at Qualtran v0.7.0, commit `bcb17bb25360ed68ce3bc084645db3e929799389`.

| Inventory | Original file | Disposition |
| --- | --- | --- |
| Q-0423 | `chemistry/trotter/__init__.py` | License-only file. The adapted implementation retains the copyright and license in its owning module. |
| Q-0424 | `chemistry/trotter/grid_ham/__init__.py` | Exports `KineticEnergy`, `PairPotential`, and `PotentialEnergy`. Closure follows the kinetic and potential file batches. |
| Q-0425 | `chemistry/trotter/grid_ham/inverse_sqrt.py` | Tables, ordered signatures and declared resources verified. The open boundaries are listed below. |
| Q-0426 | `chemistry/trotter/grid_ham/inverse_sqrt_test.py` | Unchanged suite returned 12 passes and six skips. Native table, signature, direct call and clean execution checks exercise the corresponding contracts. |

## Registers and resource declarations

The polynomial model has three ordered through ports, `x_sq`, `in_coeff[4]`, and `out`. Newton has `x_sq`, `poly`, and `target`, also through. Each port is opaque, with one logical word per array element. The cost model derives each declaration's starting width from these ports. It preserves the original child order and multiplicity.

| Declaration | Immediate children | Source model example |
| --- | --- | --- |
| Polynomial | Three real products and three signed adds, all at coefficient width | `(7, 8, 12)` gives 744 T equivalents. |
| Newton | One polynomial-width square, one scale using polynomial and distance widths, two target-width real products, one target-width signed add | `(5, 15, 22)` gives 1,161 Toffoli equivalents. |

The original tests supply scaled-coefficient cases, fixed-point arithmetic checks, and model examples. The native reference tests compare all words in each of the four banks at six width pairs, including 55 coefficient bits. They also compare both source signatures, each direct child and aggregate resources at four width triples, including the original documentation example. Address zero holds four zero words.

## Executable path and concrete boundaries

The native `grid_cubic_inverse_sqrt` compiler path retains the distance, produces a fixed-point estimate for the consumer, and erases the scratch. Its native table follows the paper's full-address segmentation, with normalized arguments, truncated Horner steps, and the Newton correction. The clean execution test uses complex amplitudes at all four-bit addresses and at selected inputs with a nonzero correction word. All scratch bits are checked through the full output state. Two deterministic seeds sample measurement histories.

The measured four-distance-bit/five-fractional-bit compute/copy/uncompute circuit has 208 scratch wires, 13 reusable measurement slots and 21,620 instructions. These include 10,284 CX, 6 CCX, 2,052 T, 2,052 T†, 1,026 measurements and 1,026 resets. These are emitted instructions for the native implementation.

Both pinned source classes raise `DecomposeNotImplementedError`. Their model
budgets cannot supply a primitive stream to compare against native execution.
The native drawer now preserves both custom root labels, the ordinary port
boxes, and each indexed coefficient box. Differential checks use the
unmodified source symbols.

The independent `GridInverseSqrtPolynomial` binding accepts four coherent
coefficient words, an argument and an arbitrary XOR target. Compiler-owned
register Horner arithmetic evaluates
`c0-x*(c1-x*(c2-x*c3))`. Argument fractional precision is explicit.
Each product is truncated at the chosen fractional position and every
intermediate wraps at coefficient width. A narrower output receives the
low bits, while a wider output receives a zero-extended word. The operation
restores its argument and coefficient registers and all borrowed work.
Reapplying it recovers an arbitrary original target.

The one-bit test covers every argument, all sixteen coefficient tuples,
both target values and a coherent spectator. Two-bit tests cover 61
coefficient/argument tuples, every target value, and a spectator, with
narrower and wider outputs. Independent integer evaluation defines every
expected amplitude. Two deterministic measurement seeds and a repeated
application check all resulting complex amplitudes and zero scratch.
The one-bit binding emits 48 T/T† and 12 measurements using six work
wires and one slot. The two-bit binding emits 336 T/T† and 84 measurements
using sixteen work wires and one slot. These are native executed counts.

The independent `GridInverseSqrtNewton` binding accepts coherent `x_sq`,
`poly`, and an arbitrary XOR target at explicit fractional widths. It
uses the existing widened Newton update and preserves the input words.
The admitted domain requires a*x*x <= 3 and a result that fits the
unsigned target. The optional coherent correction word
adds delta to 3 at the shared update's fractional scale.

The new binding tests enumerate each valid source/estimate/correction
combination at three small register formats, every occupied target value,
and a coherent spectator. Independent integer arithmetic specifies
the exact truncated output. The two-bit integer-source fixture includes
224 such inputs and emits 224 T/T† and 56 measurements using 33 work
wires and five slots. Both sampled trajectories and two-application
recovery match every complex amplitude and restore all borrowed work.

For those complete valid Newton domains and the tested coherent
polynomial domains, the shared And-channel certificate checks each
emitted compute and erase block on every basis input. It verifies that
each erased value is the exact conjunction, and that both measurement
outcomes have the same positive normalized action. This proves the
result for every measurement history on those input spans. The independent
trajectory comparisons also check their complex amplitudes. This
certificate complements the existing sampled tests of the larger
table/interpolation circuit.

The bank builder admits at least two address bits and 1–63 coefficient
bits. The source gives no arithmetic representation for every arbitrary
coefficient input or primitive polynomial/Newton decomposition. The
explicit native fixed-point convention and original resource declaration
therefore retain separate evidence. The formerly open representative complete-chain branch coverage is now proved
for both existing 4/5 and 3/16 profiles by literal And-channel composition; see
[complete source binding](GRID_INVERSE_SQRT_SOURCE_BINDING.md). The optional
coefficient conversion and original resource-only constructor capabilities are
also complete there; original opaque arithmetic ambiguity remains explicit.
