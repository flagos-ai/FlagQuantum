# Bounded fixed-point ArcTan

The compiler preserves the locked ArcTan arithmetic map. Given a
big-endian unsigned integer x, it XORs the sign of −2·atan(x)/π into
one qubit and XORs the truncated magnitude into a big-endian pure
fraction. The sign mask is zero at x=0 and one otherwise. Both
outputs may start occupied. Clean workspace is restored, so repeating
the operation clears the same masks. [Locked ArcTan source](https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/arithmetic/trigonometric/arctan.py)

This concrete path supports 1–12 input bits and 1–32 output fraction
bits. The table grows exponentially with input width. The emitter
uses the existing unary QROM with two target words and one shared
address traversal. The magnitude follows the source's truncating
QFxp conversion. All 4,096 entries at the maximum target precision
were compared with the locked method. At this precision, nontrivial
entries are at least 5.57e−5 scaled units away from an integer in
the measured binary64 evaluation. Smaller fractional widths discard
low bits of the same table.

Tests compare complete complex outputs with the original Cirq
arithmetic-gate matrix, including superpositions over occupied sign
and magnitude outputs, and execute the native IR obtained through
`circuit.append(op)`. The native primitive stream also matches the
locked QROM for the table generated from ArcTan, including measurement
feedback and clean work. All measurement histories are certified for
each tested input/output pair.

| Source bits / fraction bits | Native T/T† | CX | Measurements | Work qubits and slots each |
| --- | ---: | ---: | ---: | ---: |
| 1 / 2 | 0 | 2 | 0 | 0 |
| 2 / 3 | 8 | 22 | 2 | 1 |
| 3 / 3 | 16 | 43 | 4 | 2 |
| 4 / 6 | 48 | 149 | 12 | 3 |
| 8 / 12 | 584 | 2,462 | 146 | 7 |
| 12 / 32 | 16,376 | 118,778 | 4,094 | 11 |

Each measurement has one reset, and the table has no CCX primitive.
The locked ArcTan has a Cirq arithmetic matrix and a symbolic
`PlusEqualProduct` count, yielding a model of 8·c·p² T gates for
output fraction width p and an unspecified count c. Its own gate
decomposition is unimplemented. The native table counts are
separate from that arithmetic model.

## Polynomial-size arithmetic path

`arctan_arithmetic_ir` has the same output layout and XOR contract.
It treats zero and one exactly. For x ≥ 2 it computes r≈1/x with
the shared Newton reciprocal, squares r, then evaluates the normalized
arctangent series with the existing signed Horner circuit. Coefficients
are rounded at the working precision from a rational Machin evaluation
of π. The magnitude is 1−r·P(r²). The circuit negates the full product
word before taking the output slice, so exact-grid products receive
the correct floor. The sign mask is x≠0. No exponential table is built.
[Arctangent series and reciprocal expansion](https://dlmf.nist.gov/4.24.E4)

At working precision b and degree d, r stays in (0,1/2] on this
branch. The reciprocal error is bounded by the
[shared reciprocal estimate](REGISTER_RECIPROCAL.md). Its contribution
to the normalized angle error is at most 2/π times that bound.
The series truncation contributes at most
(2/π)·r^(2d+3)/(2d+3). Coefficient quantization and per-partial Horner
truncation are separate. For an output of p fractional bits, the final
floor adds less than 2^(−p). Bit-for-bit source parity depends on
whether the approximation crosses a quantization boundary.

| Input bits / output bits / working bits / degree | T/T† | CX | CCX | Measurements | Work qubits | Max normalized angle error | Words differing from locked ArcTan |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 / 2 / 2 / 0 | 716 | 1,646 | 6 | 179 | 46 | 0.045168 | 1 / 4 |
| 3 / 4 / 5 / 2 | 9,824 | 24,196 | 12 | 2,456 | 138 | 0.045168 | 3 / 8 |
| 4 / 5 / 7 / 3 | 20,056 | 50,152 | 17 | 5,014 | 220 | 0.023304 | 3 / 16 |
| 5 / 6 / 8 / 4 | 29,268 | 73,472 | 22 | 7,317 | 284 | 0.013626 | 5 / 32 |
| 8 / 12 / 16 / 6 | 140,820 | 358,692 | 40 | 35,205 | 770 | 0.000241 | 4 / 256 |

The counts include all uncomputation. Every differing word in these
profiles is one target ulp higher than the source. For these small
input widths, the bounded lookup uses fewer primitive gates. The
arithmetic path removes the table-size limit and remains an independent
approximation. The locked symbolic FMA count is listed above.

Tests certify all measurement histories and arbitrary initial sign
and magnitude bits in the tested configurations. They also check
complex trajectories from the IR obtained through `circuit.append(op)`
and pin the source rounding-boundary differences.
