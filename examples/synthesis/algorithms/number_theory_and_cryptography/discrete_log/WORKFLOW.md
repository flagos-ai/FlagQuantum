# Discrete logarithm interference

The native quantum function retains the source's modulus 13, element 3,
generator 7, two five-bit exponent registers and four-bit modular result.
It prepares the exponents uniformly, starts the modular result at one, applies
two existing `modular_exp` operations and reverses both QFTs. Compiler owns the
controlled modular gate design and cleanup. No host execution of quantum
arithmetic, physical wire selection or manual bitstring decoding is required.

`Output[Annotated[QNum, (5, False, 5)]]` preserves the five fractional bits
using existing standard annotation support. The modular primitive and QFT act
on the raw bit pattern. `program.probabilities` reads the same raw words as
the source adapter; `program.numeric_probabilities` reads exact fractions
`alpha/32` and `beta/32` from the verified native declarations. These are
complementary views, not a change to the circuit or implicit rescaling.

The independent Circuit constructs controlled modular permutations using
classical `pow` and uses explicit dense inverse Fourier matrices. A second
oracle builds the joint amplitude tensor analytically and applies a two-axis
FFT. Tests compare the original, native and Circuit full complex states,
absolute phase, all raw/numeric marginal keys, two further nonzero coprime
parameter pairs, bit order, actual clean work, invalid bases and budget refusal.

Both original and native use 16 physical bits, including 2 work bits,
1068 operations and depth 872. The example explicitly budgets those two work
bits; a smaller workspace budget refuses synthesis. Only the quantum program
is accepted; complete Notebook workflows and number-theory postprocessing
remain separate migration work.
