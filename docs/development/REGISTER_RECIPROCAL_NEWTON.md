# Widened fixed-point reciprocal step

The circuit evaluates one step of Newton iteration on unsigned
big-endian words. Let the source encode `a=W/2**f` and the
estimate encode `x=X/2**b`. The valid arithmetic domain is
`a >= 1` and `0 <= x <= 1/a`. The circuit XORs the
`b`-bit truncation of `2*x - a*x*x` into the result, and
preserves the source and estimate. Reapplying clears the
same result.

An exact integer square is followed by an occupied-accumulator
product subtraction. The numerator word has `n+2*b` bits,
where `n` is the source width. This holds both `W*X*X` and
`2*X << (f+b)`. After the subtraction, a single output slice
drops the low `f+b` bits. The inverse product accumulation
restores the numerator, the shifted estimate copy clears it,
and the square erases its work. The product and square share
temporary storage across their disjoint lifetimes. The
classical carry slots are also reused.
[Newton equation and truncation](https://arxiv.org/pdf/1511.08253#page=30)

For `W=3, X=2, f=0, b=3`, the update is 2. Truncating
the product before subtracting it would yield 3. Tests
compare the exact rational map for every valid input in
five layouts, certify all measurement histories, and
compare raw or sparse primitive complex amplitudes with a
coherent spectator. They verify occupied-result XOR,
workspace cleanup, and a trajectory round trip through
`circuit.append(op)`.

| Source width / source fraction / estimate width | Native T-family / CX / measurements / work wires | Classical slots |
| --- | --- | ---: |
| 1 / 0 / 1 | 32 / 69 / 8 / 9 | 2 |
| 2 / 0 / 3 | 248 / 585 / 62 / 39 | 7 |
| 3 / 1 / 3 | 352 / 857 / 88 / 53 | 8 |
| 4 / 2 / 4 | 632 / 1588 / 158 / 87 | 11 |

Counts include compute and uncompute for one application.
The locked Qualtran InvertRealNumber has no gate
decomposition. Its resource graph lists a square, a
pure-fraction multiply, an integer scale and a subtraction
per step. The pure-fraction multiplication signature
cannot directly accept the mixed-format input used here.
The graph is a source model and the table records the
native widened construction.
