# Unsigned sum of squares

The emitter binds the pinned Qualtran v0.7.0 SumOfSquares
signature for equal-width unsigned words and a result of
width `2*n + (k-1).bit_length()`. It pools the diagonal bits
and off-diagonal products of every input in the same weighted
columns and compresses each column using the shared square
network. After copying the output parity, measured erasure
clears all products and carries in reverse dependency order.
The circuit XORs the computed sum into the result and preserves
every input. A single input emits the existing Square stream.
[Appendix G, Lemmas 7–8](https://arxiv.org/pdf/2105.12767#page=78)

The pinned SumOfSquares provides its signature and Toffoli model.
It raises on gate decomposition and tensor contraction, and its
classical method returns NotImplemented. Tests independently
check the sum against integer arithmetic. The literal And and
erasure stream certifies every measurement history for all
basis inputs in layouts up to 12 input bits. The six-word
three-bit case checks equal-word inputs including the maximum
sum. The tests also verify XOR into occupied results and
round-trip cleanup, coherent forward and reverse sparse
trajectories, and every raw history in the three one-bit case.
The native append path is exercised with measured feedback.

| Input width / count | Native T-family / CX / measurements / work wires | Pinned Toffoli model |
| --- | --- | ---: |
| 1 / 3 | 4 / 18 / 1 / 1 | 1 |
| 2 / 2 | 24 / 57 / 6 / 6 | 6 |
| 2 / 3 | 32 / 95 / 8 / 8 | 9 |
| 3 / 3 | 92 / 254 / 23 / 23 | 23 |
| 4 / 3 | 172 / 478 / 43 / 43 | 43 |
| 3 / 6 | 200 / 589 / 50 / 50 | 50 |

Counts are per direction and include native cleanup. For three
two-bit inputs, three pair products and five column carries
account for the eight native logical Ands. The locked cost
formula returns nine abstract Toffolis.
