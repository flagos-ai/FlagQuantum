# Static matrix eigenphase estimation

The native qfunc prepares a private data state, allocates a typed fractional
counter, invokes `qpe(lambda: unitary(UNITARY, state), phase_result)` and drops
the data name. Compiler chooses controlled powers and inverse QFT. Users do
not design those gates, choose physical wire numbers or decode bitstrings.

All immutable matrix entries and original input amplitudes are retained.
The seven-bit matrix example retains its signed all-fractional format; the
eight-bit example retains its unsigned all-fractional format. Use
`program.probabilities` for raw words and `program.numeric_probabilities` for
exact Fraction values with the declared sign. The private data remains part
of the full statevector and is marginalized by the named probability reader.
Drop performs no reset or reuse; the environment is distinct from clean work.

The Circuit comparison constructs controlled powers independently with torch
matrix powers and a dense inverse Fourier matrix. Acceptance compares original,
native and independent complete complex states, including absolute phase,
bit order, retained data, two further nonzero amplitude vectors, exact numeric
marginals, unused-counter ownership and actual resource caps. Both examples
require zero clean work. The seven-bit example retains nine physical bits,
347 operations and depth 322; the eight-bit example retains eleven physical
bits,26898 operations and depth23711. No gate-count improvement is claimed.

The initial native qpe scope is one static unitary call in a zero-argument
lambda at the unconditional root, with an allocated and unused whole
one-to-eight-bit all-fractional counter. Arbitrary callbacks, flexible powers,
nested or inverse QPE and numeric-to-array helper borrowing remain separate.
Only the quantum program is accepted; full Notebook workflows remain pending.
