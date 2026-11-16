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

Both pinned source classes raise `DecomposeNotImplementedError`. Their model budgets cannot supply a primitive stream to compare against native execution. Their custom drawer labels and acceptance of arbitrary caller-supplied coefficient arrays have no equivalent in the fixed-table native owner. The bank builder admits at least two address bits and 1–63 coefficient bits. Complete enumeration of the physical circuit's measurement histories remains open.
