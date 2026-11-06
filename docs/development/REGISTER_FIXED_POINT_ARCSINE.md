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
