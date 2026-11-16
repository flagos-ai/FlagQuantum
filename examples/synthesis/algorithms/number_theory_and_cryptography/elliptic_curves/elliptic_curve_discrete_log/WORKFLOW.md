# Elliptic-curve discrete logarithm

Run from the repository root:

```sh
python -m examples.synthesis.algorithms.number_theory_and_cryptography.elliptic_curves.elliptic_curve_discrete_log.main
python -m examples.synthesis.algorithms.number_theory_and_cryptography.elliptic_curves.elliptic_curve_discrete_log.comparisons.circuit
```

The program retains the curve y²=x³+5x+4 over F_7, generator (0,5), seed
(4,2), two uniform three-bit scalar registers and both inverse Fourier
transforms. `build_program((3,2))` selects the second independently validated
source seed. Named scalar words decode as word/8, matching the source's
fractional format. The point word is x+8*y; the executable prints decoded
coordinates and fractional scalar values.

Classical finite-field group arithmetic constructs a translation permutation
for each of G, 2G and 4G. All ten group elements are included: nine affine points
and an infinity sentinel encoded as (7,7). Non-curve words are fixed. This
explicitly extends the original's unproved exceptional affine branches; the
retained source inputs remain in nonsingular cosets. Each permutation is
decomposed into cycles and Gray-path basis transpositions, then composed in
native qfunc using checked controls and within/apply masks. No final state is
precomputed, no six-bit dense matrix bypasses the five-bit unitary limit, and
no source loader participates in the native program.

The independent Circuit computes affine scalar sums directly and encodes the
result with XOR gates before analytic inverse DFT matrices. It does not use the
native Gray decomposition. Acceptance compares complete source/native/reference
states for both seeds, adds signed nonuniform coherent scalars, verifies every
translation word (including infinity and invalid words) on a distinct complex
input, checks inverses and hidden workspace, and exercises overlap, seed and
budget refusals. The original source and baseline remain unchanged.

Native default resources are 15 physical bits, three clean workspace bits,
5,010 operations and depth 4,995. Original resources are 14 bits, two workspace
bits, 12,192 operations and depth 7,233. The basis differs: the native circuit
includes Toffoli operations, so raw operation counts are not an equal-basis
hardware performance comparison. The full-state source error is below 2e-13.
This verifies the quantum program; the complete Notebook workflow is separate.
