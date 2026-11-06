# Fixed-point Horner multiply-add

The register component accumulates one Horner step into an occupied
result, modulo the result width. The big-endian argument is unsigned
and has an explicit fractional width f. The previous word, result
and compile-time signed coefficient share the same two's-complement
format. The coefficient is specified in result ulps.

For argument word A and previous-word bit j, the circuit adds the
truncation of A·2^(j−f) if that bit is set. The most significant
bit instead subtracts its truncated magnitude. Slices drop both
underflow and overflow columns before each controlled add. This
implements the per-partial truncation of Appendix A and the
multiply-accumulate step of Appendix B. [Appendices A and B](https://arxiv.org/pdf/1805.12445#page=7)

The adjoint first subtracts the coefficient and then reverses the
controlled-add sequence with opposite arithmetic. Source, previous
word and initially clean work are restored in either direction.
The same load and carry bank serves every contribution.

For A=3, f=2 and a three-bit previous word, +3 produces 1 and −1
produces −2 under per-partial truncation. Flooring the full products
would give 2 and −1, respectively. Tests compare all source pairs in
six layouts, with occupied outputs, modular wraparound, complex
amplitudes and all measurement histories. The one-bit case also
forces the two raw measurement outcomes and checks a forward and
adjoint `circuit.append(op)` round trip.

| Argument bits / signed bits / argument fraction / coefficient | Native T/T† | CX | Measurements | Work wires and slots each |
| --- | ---: | ---: | ---: | ---: |
| 1 / 1 / 0 / −1 | 4 | 8 | 1 | 1 |
| 2 / 3 / 2 / −4 | 36 | 76 | 9 | 5 |
| 4 / 4 / 3 / 1 | 100 | 213 | 25 | 7 |
| 5 / 5 / 3 / −2 | 164 | 354 | 41 | 9 |
| 8 / 8 / 5 / 1 | 424 | 934 | 106 | 15 |

Each direction has the same counts in this table. Each measurement
has one reset, and CCX is absent from the emitted primitive stream.
The paper's fixed-width FMA model estimates 49 Toffolis at n=4,p=1
and 177 at n=8,p=3 using its own adder choices. The native counts
use shared measured Add and CAdd. Qualtran v0.7.0 does not provide
an executable Horner step for its ArcSin model. The Add, AddK and
CAdd leaf implementations retain the previously validated source
decompositions. [Appendix B resource model](https://arxiv.org/pdf/1805.12445#page=7)
