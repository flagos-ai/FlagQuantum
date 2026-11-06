# Nonnegative fixed-point arcsine

The compiler consumes an unsigned big-endian input x in [0, 1], with
one integer bit. The result also has one integer bit and is updated by
XOR. The caller supplies the quantized π/2 word in result ulps and the
coefficients of P in descending order. The polynomial word has one
sign bit, one integer bit and p fractional bits. Its evaluated value
must be nonnegative over the reduced input domain. The high branch's
scaled product must fit the supplied π/2 word. These are data-domain
preconditions on the caller's approximation. [Appendix D](https://arxiv.org/pdf/1805.12445#page=10)

Below 1/2 the circuit evaluates x·P(x²). At and above 1/2 it evaluates
π/2 − 2·sqrt(z)·P(z), with z=(1−x)/2. A coherent flag selects the
multiply factor and the high branch's scale and correction. Conditional
XOR into clean factor storage implements the selection. One live
unsigned product supplies both output alignments before its columns
are erased. On the high branch the circuit truncates the doubled
product before subtracting from the quantized π/2. Source, branch
flag, intermediate registers and scratch are restored. The selected
factor has max(input fraction, working precision) fractional bits.

Tests cover every legal input in five layouts, including the endpoints
and the 1/2 boundary, and C1 and C2 settings. They verify occupied
results and every measurement history with a channel certificate.
Coherent sparse trajectories execute the primitive IR obtained through
`circuit.append(op)` and compare both branches with an entangled spectator.

| Input frac / polynomial frac / output bits / precision / updates | Native T/T† | CX | CCX | Measurements | Work wires | Max error on encoded inputs (rad) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 / 1 / 3 / 1 / 0 | 372 | 810 | 50 | 93 | 31 | 0.07080 |
| 2 / 2 / 5 / 2 / 1 | 6,856 | 16,918 | 88 | 1,714 | 196 | 0.21444 |
| 3 / 4 / 8 / 4 / 1 (C2) | 16,896 | 43,250 | 136 | 4,224 | 442 | 0.17171 |
| 4 / 4 / 12 / 3 / 1 | 21,824 | 55,406 | 170 | 5,456 | 486 | 0.14518 |

These counts include the complete cleanup and keep CCX separate. Each
measurement has one reset. The corresponding coefficient words are
(2), (1, 4), (3, 16) and (1, 3, 16). They are test approximations.
The paper describes a minimax approximation but gives no coefficient
table in the cited sections. The error column applies to these
specific coefficients and working precisions. The signed wrapper
and a reproducible minimax coefficient set are separate work.

Qualtran v0.7.0's ArcSin provides a cost model and a classical method.
Its resource estimate includes the whole algorithm and uses different
adder and precision assumptions. Its classical method may return an
out-of-range word, as documented in the
[range reduction evidence](REGISTER_ARCSINE_REDUCTION.md).
[Locked ArcSin source](https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/arithmetic/trigonometric/arcsin.py)

## Signed input and output

The signed wrapper accepts big-endian two's-complement input and
output words with one sign bit and one integer magnitude bit. The
input domain is [−1, 1]. It preserves the original sign and
copies the remaining input bits into a clean magnitude word.
On negative inputs it conditionally complements and increments
the magnitude. This obtains the exact magnitude even at −1.
It then evaluates the unsigned kernel and conditionally negates
the signed angle. The angle is XORed into the caller's result,
and the exact adjoints clear both magnitude and angle. The
unsigned kernel runs twice to compute and clear the angle.
[Appendix D sign handling](https://arxiv.org/pdf/1805.12445#page=10)

The tests check every encoded input in three layouts, including
±1, zero and ±1/2, and verify exact odd symmetry of the quantized
output. They also certify all measurement histories and check
cross-sign complex amplitudes with entangled spectator states
using the IR from `circuit.append(op)`.

| Input frac / polynomial frac / signed output bits / precision / updates | Native T/T† | CX | CCX | Measurements | Work qubits |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 / 1 / 4 / 1 / 0 | 776 | 1,740 | 100 | 194 | 37 |
| 2 / 2 / 6 / 2 / 1 | 13,768 | 34,044 | 176 | 3,442 | 205 |
| 3 / 4 / 9 / 4 / 1 (C2) | 33,880 | 86,825 | 272 | 8,470 | 455 |

The signed counts use the same coefficient words as their unsigned
counterparts and include cleanup. The measurement increment over
two unsigned calls is twice the sum of the lower-bit carry counts
of the magnitude and angle. Qualtran v0.7.0 still exposes the
unsigned-format ArcSin resource model and no executable gate flow
for the signed wrapper.
