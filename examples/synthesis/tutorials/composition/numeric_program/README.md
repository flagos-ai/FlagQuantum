# Exact numeric composition

Declare quantum formats and write the arithmetic; synthesis constructs the gates.
The source `x` is a signed one-bit value with one fractional bit, so the prepared
superposition contains 0 and -1/2. Two reusable qfunc helpers compute
`doubled = 2*x` and `result = doubled + 0.5`. Output sign, width and binary scale
are inferred and propagated across the calls. `Annotated[QNum, (...)]` is the
static-type-checker-friendly form of `QNum[width, signed, fractional_bits]`.

```sh
python -m examples.synthesis.tutorials.composition.numeric_program.main
python -m examples.synthesis.tutorials.composition.numeric_program.comparisons.circuit
```

The decoded distribution contains `(0, 0, 1/2)` and `(-1/2, -1, -1/2)`, each
with probability 1/2. Fractions are returned directly by
`program.numeric_probabilities`; no physical wires or bit-string decoder are
needed. The phase operation also creates distinct complex amplitudes, including
a constant global phase. The independent Circuit comparison checks their
absolute values and phases and verifies all arithmetic scratch returns to zero.

Arithmetic is exact: a decimal such as `0.1` is refused rather than silently
rounded. This additional FlagQuantum tutorial does not award an upstream corpus
or Notebook workflow acceptance.
