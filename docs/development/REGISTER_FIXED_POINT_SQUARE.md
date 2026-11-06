# Square an unsigned fixed-point fraction

The native emitter takes one big-endian fraction and XORs its
truncated square into an equal-width fractional result. The
same circuit erases a matching result. It preserves the input
and returns all partial-product and carry wires to zero.

The term set follows the written equations D24 and D28. In
one-based fractional positions, the odd case retains diagonal
terms through `(d-1)//2` and pairs with `i+j <= d`. The
even case omits the midpoint diagonal and retains pairs with
`i+j <= d+1`, including the last-row pair from D28 and D30
that is not shown in the displayed D29 example. The shared
weighted-column emitter reduces those terms and erases them
with measurement feedback. [Appendix D6](https://arxiv.org/pdf/2007.07391#page=73)

The locked Qualtran v0.7.0 SquareRealNumber signature includes
two inputs, although its equation and the cited construction
take one. The locked member supplies a cost model. Its gate
and tensor methods raise, and its classical method returns
NotImplemented. The native single-input interface follows the
paper equation. Tests preserve the signature discrepancy and
check an independent ordered-sum oracle for the numerical map.

The tests compare the literal product And stream, certify every
history for all valid inputs at widths 3, 4, 5, 6, 8 and 10,
and check occupied-result XOR in both directions. Widths 3
and 4 project every raw history with a coherent spectator.
Widths 5 and 8 run sparse primitive trajectories. An append
round trip uses the local trajectory runtime.

| Width | Native T-family / CX / measurements / work wires | Pinned Toffoli model |
| ---: | --- | ---: |
| 3 | 8 / 16 / 2 / 2 | 0 |
| 4 | 28 / 60 / 7 / 7 | 4 |
| 5 | 28 / 68 / 7 / 7 | 8 |
| 6 | 60 / 148 / 15 / 15 | 14 |
| 8 | 116 / 296 / 29 / 29 | 28 |
| 10 | 184 / 480 / 46 / 46 | 46 |

Native counts apply per direction and include measurement
erasure. The source model uses `d*d//2-4`.
