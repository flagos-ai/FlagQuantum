# Single Montgomery multiplication window

The register emitter takes a little-endian multiplier window and
big-endian operand, accumulator and lookup index. The index starts at
zero and remains a logical output. The source stages are controlled
addition on successive accumulator slices, copy of the low bits,
correction lookup, addition of that correction and bank cleanup.
The final source rotation is an output wire order. For the adjoint,
the caller supplies that rotated input order and the matching index.

The table uses the source concrete block choice and the correction
`(-index * p^-1 mod 2^w) * p` for an odd modulus. The emitter borrows
vacant bank wires during controlled addition. A separate n+w-1 wire
pool serves the correction adder and measured lookup, with 2n classical
slots for controlled addition. All bank and work wires end at zero.
Forward and adjoint expose the source logical accumulator order.

The gate oracle walks the entire locked SingleWindowModMul and adjoint
Soquet graphs. It compares the original controlled add, XOR, add,
lookup and wire-bookkeeping stages. At the measurement-adjoint leaf it
checks the original wrapper's bank packing and binds the executable
reverse of the accepted forward components. This reverse has its own
literal resource counts.

The numerical oracle calls the original classical composite in both
directions. Tests certify every And history for unblocked cases and
exhaust all eight raw histories for the one-bit case. Larger circuits
use complete occupied-vector comparisons for seeded sparse trajectories.
The 54-wire blocked case covers all 32 lookup indices and additional
carry edges, with a coherent spectator. Every tested branch restores
bank and temporary work, with the index restored to zero after the
adjoint.

On the carry-safe domain `2*p-1 < 2^n`, `y < p` and
`accumulator < 2*p`, the tested integer recurrence agrees with the
original gate graph. The emitted arithmetic keeps the source
wraparound behavior. At n=2, w=1, p=3, x=1, y=2 and accumulator=5,
the original graph yields logical accumulator 1 while its high-level
classical method gives 5. The test preserves that witness.

| n, w, p | Physical qubits | Literal T-family | Literal CX | Literal measurements | Source model T | Source model measurements |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 3, 1, 3 | 16 | 36 | 88 | 9 | 36 | 9 |
| 4, 2, 5 | 25 | 100 | 239 | 25 | 92 | 23 |
| 5, 5, 13 | 54 | 428 | 1,116 | 87 | 364 | 73 |

Literal counts apply per direction. Model T uses the source's default
T-equivalent accounting. The difference is the literal executable bank
clear in place of the measurement-adjoint leaf estimate. The physical
qubit count agrees with the source model for all tested cases.

Six module cases and 203 dependency cases passed. The emitter reaches
100% statement coverage. The locked mod_multiplication.py blob matches
the inventory.
