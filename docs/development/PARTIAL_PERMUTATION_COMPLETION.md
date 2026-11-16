# Complete partial permutations in sparse preparation

The pinned Qualtran v0.7.0 `linalg/permutation.py` at
`bcb17bb25360ed68ce3bc084645db3e929799389` has source blob
`2655f6e61242309a9691afaba5171e324c2cff0c`. Its original tests have blob
`9c5127321439bd8e2147924a94ec320261bf5260`. The frozen raw traversal and
its explicitly source-named native adapter still reproduce the original
insertion-order behavior.

For the valid injection `{0: 1, 1: 3, 2: 0}`, the source yields
`(0, 1, 3)` and omits `2 -> 0`. The native production path validates the
finite domain and injectivity, then follows predecessors to the actual head
before walking each path. It closes the example as `(2, 0, 1, 3)`. Closing
each path independently minimizes transpositions by maximizing the number
of disjoint cycles. Unbound states outside those paths remain fixed.
Components retain first encounter order and existing cycles retain their
starting vertex and orientation. The same forward traversal also serves
the explicit source adapter.

An independent transposition BFS verifies minimality over all 1,313
ordered injections on four vertices. The 1,121 cases where the source
already realizes every edge preserve its exact cycle sequence. The 192
source failures are repaired in the production path. The original source
docstring's unsupported `N` keyword remains historical evidence. The native
entry point accepts an explicit size for validation.

The sparse preparation consumer now uses the complete injection so that
arbitrary valid index order preserves the intended complex state. The
previous `(3, 0)` witness appends a full preparation on a two-bit target and
a three-bit phase gradient. All 1,024 unnormalized measurement branches
match the complete expected state, including the restored phase gradient
and zero work wires. Exact comparison uses a three-bit phase grid to
represent both component phases and their average. The singleton
and symbolic preparation cases remain under Q-0697.

| Case | Measured physical stream | Independent model |
| --- | --- | --- |
| Corrected four-cycle | CX39, X20, H15, T10, Tdg10, S5, measure5, conditional CZ5, reset5. Two clean work wires and one classical slot. | The pinned arithmetic `Permutation` on the explicit corrected cycle reports AND5, Clifford14, measurement5, and logical width 4. |
| Corrected sparse append | CX98, H30, X22, T20, Tdg20, S10, measure10, conditional CZ10, reset10, RX2. Six clean work wires and two classical slots. | Native analytical model: Toffoli2, AND6, Clifford30, measurement6, and peak logical width 10. |

The four-cycle test compares the exact gates, operand order, measurement
feedback, and resource declarations against the pinned arithmetic
`Permutation` on the explicit valid cycle. It checks all 32 forward and
32 inverse branches with an entangled spectator and zero work.
