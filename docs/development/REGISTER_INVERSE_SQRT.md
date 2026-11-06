# Fixed-point inverse square root

The compiler composes the untuned Appendix C seed (C1) with an explicit
number of widened Newton updates. Big-endian unsigned inputs have f
fractional bits. Each estimate has floor(f/2)+1 integer bits and b
working fractional bits. The result may keep any prefix that contains
all the integer bits. It is XORed into the caller's result register.
Zero input has a zero mask. [Appendix C](https://arxiv.org/pdf/1805.12445#page=8)

The smallest leading-bit exponent determines the smallest selected
power 2^k. Requiring b ≥ 1−k ensures the seed remains positive after
truncation for every positive input. The seed and all successive
estimates are stored in separate history words. The seed and updates
share one scratch bank. After the result is copied, the updates and
seed are applied in reverse order at the component level. Each
component XORs its output, so the original measured cleanup stays
valid. Work starts and ends at zero and classical slots are reusable.

For a positive input, the exact C1 seed lies between 1/√(2a) and
1/√a. Every truncated estimate is nondecreasing and at most 1/√a.
For the untruncated iteration with normalized error e=1−x√a, the
next error is (3e²−e³)/2. On the underestimation interval the
update's absolute derivative is at most 3/2. After s updates, the
difference between the truncated and untruncated estimates is at
most 2^(-b)·sum((3/2)^j, j=0..s). Shortening the result adds less
than one result ulp.

Tests compare every source word in five layouts against exact rational
arithmetic, including occupied outputs and the zero-input convention.
They also check the literal component sequence, every measurement
history via a channel certificate, and raw primitive trajectories for
coherent input entangled with a spectator. The zero-update configuration
runs twice through `circuit.append(op)` to restore its input.

| Source bits / source fraction / working fraction / iterations / output bits | T/T† | CX | CCX | Measurements | Work wires | Classical slots |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 / 0 / 1 / 0 / 1 | 32 | 73 | 4 | 8 | 8 | 2 |
| 1 / 0 / 1 / 1 / 2 | 312 | 666 | 4 | 78 | 32 | 4 |
| 2 / 2 / 2 / 2 / 3 | 2,688 | 6,483 | 8 | 672 | 109 | 9 |
| 3 / 1 / 2 / 2 / 2 | 2,368 | 5,644 | 12 | 592 | 77 | 8 |
| 5 / 3 / 4 / 3 / 5 | 12,472 | 31,845 | 20 | 3,118 | 256 | 16 |

Native counts include the complete reversal. Each measurement has one
reset. The paper estimates the fixed-width untuned seed and forward
Newton iterations, before reversing the history. For instance its
model with uniform width 8, 3 integer bits and 2 updates gives
1,912 Toffolis. This estimate uses a different arithmetic width and
cleanup scope from the native table. Qualtran v0.7.0 has no gate
decomposition or independent model for this C1 iteration wrapper.
The chemistry update uses an integer source and a distinct resource
graph, recorded in [the Newton update evidence](REGISTER_INVERSE_SQRT_NEWTON.md).
The tuned C2 initialization and multiplication back to obtain the
square root remain separate.
[Appendix C resource formulas](https://arxiv.org/pdf/1805.12445#page=9)
