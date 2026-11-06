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
Within the unsigned kernel, it selects the low product or corrected
high angle while both are live. It conditionally negates the selected
angle, XORs it into the caller's result, and reverses the negation and
selection before the kernel's own cleanup. The exact adjoint then
clears the input magnitude. The consumer reuses scratch beyond the live
product, including the carry area after the correction. The default
nonnegative emitter retains its original primitive gate order.
[Appendix D sign handling](https://arxiv.org/pdf/1805.12445#page=10)

The tests check every encoded input in four layouts, including
±1, zero and ±1/2, and verify exact odd symmetry of the quantized
output. They also certify all measurement histories and check
cross-sign complex amplitudes with entangled spectator states
using the IR from `circuit.append(op)`.

| Input frac / polynomial frac / signed output bits / precision / updates | Native T/T† | CX | CCX | Measurements | Work qubits |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 / 1 / 4 / 1 / 0 | 404 | 930 | 56 | 101 | 37 |
| 1 / 1 / 20 / 1 / 0 | 660 | 1,778 | 120 | 165 | 110 |
| 2 / 2 / 6 / 2 / 1 | 6,912 | 17,126 | 98 | 1,728 | 205 |
| 3 / 4 / 9 / 4 / 1 (C2) | 16,984 | 43,575 | 152 | 4,246 | 455 |

The signed counts use the same coefficient words as their unsigned
counterparts and include cleanup. Relative to commit `c3824495`, one complete unsigned call is
saved. At 2 / 2 / 6 / 2 / 1 this removes 6,856 T/T† and 1,714
measurements, with the same 205 work qubits. The 20-bit output
case removes 500 T/T† and 125 measurements while requiring 21
extra work qubits, because the sign correction needs a larger
clean carry bank while the product remains live. Qualtran v0.7.0 still exposes the
unsigned-format ArcSin resource model and no executable gate flow
for the signed wrapper.

## Reproducible coefficient profile

The paper identifies its approximation method and domain. The cited
sections and the locked Qualtran source do not give its numerical
coefficient table. `arcsine_polynomial_coefficients` constructs an
independent, rational Chebyshev projection for
`asin(sqrt(z))/sqrt(z)` on [0, 1/4], with value one at z=0.
It uses no external solver or floating-point arithmetic.
[Paper, Sections IV and VI and Appendix D](https://arxiv.org/pdf/1805.12445)
[Locked ArcSin source](https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/arithmetic/trigonometric/arcsin.py)

The series coefficient of z^n is binomial(2n,n)/(4^n·(2n+1)).
Under z=(1+cos θ)/8, the powers of z have nonnegative Chebyshev
coefficients. Their dropped sum bounds the projection error because
each Chebyshev polynomial has magnitude at most one. The omitted
Taylor series is bounded by 1/(3·4^N) after order N. The routine adds
the monomial quantization bound, rounding each coefficient to
nearest with ties to even. These bounds cover the continuous factor
before reversible arithmetic. The approximation is a Chebyshev
projection and does not claim the paper's minimax coefficients.

| Degree | Coefficient fractional bits | Uniform factor bound |
| --- | ---: | ---: |
| 3 | 24 | 1.661e−6 |
| 4 | 18 | 5.387e−7 |
| 4 | 24 | 1.114e−7 |
| 5 | 24 | 1.231e−8 |

For example, degree four with 12 coefficient fraction bits gives
(207, 163, 309, 683, 4096), in descending order. With three input
fraction bits, a ten-bit signed output, seven square-root working
fraction bits and two Newton updates, the maximum error over all
17 encoded signed inputs is 0.008463 rad. The factor bound before
arithmetic is 2.742e−5, and the maximum observed difference between
the rounded polynomial and its per-partial Horner circuit is
0.000307 in factor units. Square-root, product and output truncation
contribute to the final angle error.

That profile emits 60,688 T/T†, 153,766 CX, 166 CCX and 15,172
measurements with 765 work qubits and 27 classical slots, including
cleanup. Tests independently enclose the exact factor at rational
points, check the full encoded signed domain and all measurement
histories, and execute representative coherent amplitudes from the
IR produced by `circuit.append(op)`.
