# Fixed-point square root

Appendix C obtains the square root by multiplying the source by its
inverse-square-root estimate. This compiler component consumes the last
live Newton history word, computes the exact unsigned column product
and XORs the requested slice into the result. Product workspace reuses
the clean iteration scratch and is erased before the inverse-square-root
history is uncomputed. [Appendix C, square root](https://arxiv.org/pdf/1805.12445#page=9)

The source is a big-endian unsigned word with n bits and f fractional
bits. The result has ceil((n−f)/2) integer bits and uses its remaining
bits for the fraction. This includes pure fractional source and result
words. An output finer than the exact product's fractional width adds
constant zero low bits to the mask. Zero input leaves the result
unchanged. All clean caller-owned workspace returns to zero. Optional C2 tuning
requires at least one Newton update before the product.
[C2 evidence](REGISTER_INVERSE_SQRT_SEED.md)

If the inverse estimate is x and the input is a>0, the error in the
unrounded product is a·(1/√a−x). Truncating the product adds less than
one result ulp. The circuit retains the entire working estimate until
the product, so a shorter result does not truncate x prematurely.

Tests exercise every input in six layouts, including occupied outputs,
pure fractions and results with zero-padded low bits. They compare the
literal gates before and after the multiply-back with the existing
inverse-square-root stream, certify every measurement history, compare
primitive complex amplitudes with an entangled spectator, and run a
`circuit.append(op)` round trip.

| Source bits / source fraction / working fraction / iterations / output bits | Total T/T† | Total CX | Total CCX | Total measurements | Total work wires | Added T/T† / measurements / work wires |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 / 0 / 1 / 0 / 1 | 40 | 85 | 4 | 10 | 8 | 8 / 2 / 0 |
| 2 / 2 / 2 / 1 / 2 | 1,456 | 3,482 | 8 | 364 | 105 | 48 / 12 / 0 |
| 3 / 1 / 2 / 2 / 2 | 2,428 | 5,770 | 12 | 607 | 77 | 60 / 15 / 0 |
| 4 / 3 / 3 / 0 / 9 | 524 | 1,193 | 16 | 131 | 40 | 140 / 35 / 15 |
| 5 / 0 / 4 / 0 / 3 | 788 | 1,791 | 20 | 197 | 50 | 180 / 45 / 19 |

Each measurement has one reset. Added counts compare with the
inverse-square-root operation at the same working precision and
iteration count. The product reuses the existing classical slots.
The paper models the increment as one multiplication in its uniform
fixed-point circuit. Qualtran v0.7.0 has no standalone decomposition
for this Appendix C path.
