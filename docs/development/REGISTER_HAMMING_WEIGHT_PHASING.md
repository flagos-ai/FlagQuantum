# Direct Hamming-weight phasing

The shared emitter follows the locked Qualtran v0.7.0
HammingWeightPhasing graph. It computes the Hamming weight into
caller-owned clean count and junk words, applies powers of Z on
the count, and runs the original measured count-tree adjoint.
The phase order is least significant first, with exponents
`exponent * 2^i`. The adjoint reverses that phase order and negates
the exponents. A basis word x acquires phase
`exp(i π exponent popcount(x))`. Both the count and junk words
return to zero.

The gate oracle reads the source graph, the original forward
count generator and its measured adjoint. It compares gates,
operands and feedback per wire, as the source graph can commute
operations on distinct wires. It also compares the exact phase
sequence. Source epsilon is divided across the count phases.
The native phase instructions carry radians directly.

For a one-bit input, the pinned graph exposes the correct count
and phase but its adjoint adapter cannot bind the zero-width junk.
The native route applies the inverse of the source forward
operation, which is one self-inverse CX. The recursive source
qubit estimate also fails for that layout.

The tests use full complex vectors over all basis words in each
tested width and a coherent spectator. They check every raw
measurement history through four bits, complete sparse trajectories
for larger words, and all-history And cleanup certificates.

| Source bits | Physical wires | Clean count and junk | Literal phases | Literal T-family | Literal CX | Literal measurements | Source model And / rotations / measurements |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3, exponent -0.375 | 6 | 3 | 2 | 4 | 20 | 1 | 1 / 2 / 1 |
| 4, exponent 0.125 | 10 | 6 | 3 | 12 | 50 | 3 | 3 / 3 / 3 |
| 8, exponent 0.625 | 19 | 11 | 4 | 28 | 114 | 7 | 7 / 4 / 7 |

The native resource counts are before phase synthesis. The source
model repeats a representative rotation, while the native IR
retains each original phase angle.

Run `python -m examples.algorithms.register_hamming_weight_phasing`
to verify the phase and clean work through `circuit.append(op)`.
