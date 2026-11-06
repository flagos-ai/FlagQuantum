# Unsigned integer square

The pinned Qualtran v0.7.0 Square and its adjoint supply the
classical and tensor contracts. Both raise
`DecomposeNotImplementedError` on gate decomposition. The native
emitter implements the column construction from Appendix G.
Diagonal bits feed column `2i`. Products of distinct bits feed
column `i+j+1`. Shared reversible compressors reduce each column
and pass the carries to the next one. [Appendix G, Lemma 7](https://arxiv.org/pdf/2105.12767#page=76)

The result is the XOR of the square with a caller-owned
double-width register. A zero result implements Square. An
already computed result implements the adjoint on its valid
input subspace. The circuit copies the parity bits and erases
the work in reverse dependency order with measured And erasure.
A single classical slot serves all immediate feedback steps.

All valid inputs from one through eight bits are checked in both
directions by composing exact And and erasure certificates for
every history. The one- and two-bit tests project every raw
history. The three-bit tests project three raw histories. The
raw tests use both pinned tensors, including a coherent
spectator, and compare complex amplitudes without phase
normalization. A two-bit square and erase also run through
`circuit.append(op)` and the trajectory runtime.

| Input bits | Native T-family | Native CX | Native measurements | Work wires | Pinned Toffoli model |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0 | 1 | 0 | 0 | 0 |
| 2 | 8 | 17 | 2 | 2 | 2 |
| 3 | 24 | 55 | 6 | 6 | 6 |
| 4 | 48 | 115 | 12 | 12 | 12 |
| 8 | 224 | 575 | 56 | 56 | 56 |

Each off-diagonal product or carry uses a four-T And with a
measured erasure. The two groups contribute equally to
`n(n-1)` And invocations. The native counts include cleanup
and apply per direction. The pinned model reports abstract
Toffolis and omits measurement, reset and Clifford costs.
