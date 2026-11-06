# Reversible fixed-point Horner polynomial

The compiler evaluates coefficients in descending degree order. The
big-endian argument is unsigned with an explicit fractional width.
Each coefficient is a signed integer in result ulps. All intermediate
words use the result's two's-complement format and the same width.
Each Horner step preserves the per-partial truncation and modular
wraparound of the [fixed-point multiply-add](REGISTER_FIXED_POINT_HORNER.md).
[Appendix B](https://arxiv.org/pdf/1805.12445#page=7)

The circuit loads the leading coefficient, computes each step into a
clean history word, XORs the last word into the result, then invokes the
exact step adjoints in reverse order. The coefficient load is finally
erased. All steps reuse one load and carry bank. The caller provides
clean work and measurement slots. A constant polynomial uses direct
XOR and needs no scratch or measurement.

At x=3/4, coefficients (1, 1, −4) and a three-bit result, the
per-step circuit returns word 4. Evaluating the exact polynomial
then flooring and reducing modulo eight gives word 5. Tests check
every argument and every occupied result in five layouts, literal
forward and reverse order, all measurement histories by certificate,
complex amplitudes and work cleanup. They force all four histories
of the smallest measured configuration and exercise
`circuit.append(op)` through a compute/erase round trip.

| Argument bits / result bits / argument fraction / degree | T/T† | CX | Measurements | Work wires | Classical slots |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 / 1 / 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| 1 / 1 / 0 / 1 | 8 | 17 | 2 | 3 | 1 |
| 2 / 3 / 2 / 2 | 144 | 307 | 36 | 14 | 5 |
| 4 / 3 / 1 / 3 | 344 | 681 | 86 | 17 | 5 |
| 5 / 4 / 4 / 4 | 776 | 1,642 | 194 | 27 | 7 |
| 8 / 8 / 5 / 4 | 3,392 | 7,480 | 848 | 55 | 15 |

Native counts include the complete reverse history. Each measurement
has one reset, and no CCX remains in the primitive stream. Resource
counts depend on the encoded coefficients. Appendix B models forward
evaluation as d fixed-width multiply-accumulates, for instance 708
Toffolis at n=8, p=3 and d=4 with its own adders. Qualtran v0.7.0
provides no executable Horner component for ArcSin. Its full
ArcSin model and the choice of approximation coefficients remain
separate from this reusable circuit.
