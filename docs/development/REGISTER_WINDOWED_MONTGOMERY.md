# Windowed Montgomery multiplication with retained history

The register emitter composes the locked Qualtran v0.7.0
`DirtyOutOfPlaceMontgomeryModMul` forward and reverse graphs. It binds
big-endian operands and processes multiplier windows from least to most
significant. Each index word is big-endian, and the index words are retained
in window order. The caller owns the zero-initialized target, index, flag,
padding, bank and work registers.

The source allocates w high accumulator bits before the first window. After
all window rotations, the first w wires of the original target are zero.
The last n wires of the rotated accumulator are the logical result. This
aliases the tail of the original target followed by the padding. The
adjoint consumes that exact output layout and returns the original target
order, with the entire target, index, reduction flag, padding, bank and work
cleared. The compiler shares the clean bank and work pool between windows,
the comparison and the controlled constant addition. It reuses 2n classical
slots.

The source reduction computes `reduced = 1 XOR (target < p)`, then
conditionally **adds p**. The decomposition is preserved. For n=5, w=5,
p=13, x=5, y=9, the original decomposed graph produces target=27,
indices=13 and reduced=1, while the high-level classical method reports
target=1. For n=5, w=1, p=9, x=y=3, the original graph produces target=18
and the high-level method reports zero. The source adjoint reverses the
decomposed values in both cases. These witnesses prevent treating the
source decomposition as a canonical modular product.

The original public wrapper's Soquet graph supplies allocations, frees,
operand order and component connections in both directions. The oracle
compares emitted gates per wire, including measurement feedback, with
the original executable components. The comparator uses the source
Cirq self-inverse decomposition. The QROAM measurement-adjoint leaf
has a source resource model. The native path clears the bank by exactly
reversing its forward components and accounts for those gates separately.

Tests cover every residue pair, including zero, for all five layouts.
Complete sparse complex vectors include a coherent spectator and check
the actual output and the adjoint restoration, including zero work.
An exact And-channel certificate covers every measurement history for
the layouts without blocked lookup exchanges. The blocked case checks
two seeded complete trajectories per direction. The small single-window
primitive also retains its exhaustive raw-history tests.

| n, w, p | Physical qubits | Literal T-family | Literal CX | Literal measurements | Source model T | Source model measurements |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 3, 1, 3 | 21 | 128 | 315 | 32 | 128 | 32 |
| 4, 2, 5 | 30 | 228 | 549 | 57 | 212 | 53 |
| 5, 1, 9 | 33 | 336 | 831 | 84 | 336 | 84 |
| 5, 5, 13 | 55 | 464 | 1,210 | 96 | 400 | 82 |
| 6, 2, 7 | 42 | 464 | 1,125 | 116 | 440 | 110 |

Literal counts apply per direction. Model T uses the source default
T-equivalent convention. The physical qubit counts agree with the
source model. The gate and measurement deltas are accounted for by
the executable bank cleanup in each window.

Run `python -m examples.algorithms.register_windowed_mod_mul` to bind
the forward and reverse operations through `circuit.append(op)` and
inspect the physical output layout and primitive counts.
