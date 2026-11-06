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
