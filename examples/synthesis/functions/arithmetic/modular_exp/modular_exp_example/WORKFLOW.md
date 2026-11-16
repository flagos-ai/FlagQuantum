# Coherent modular powers

The source Notebook initializes a three-bit target to one, prepares a uniform
three-bit exponent, applies modular exponentiation with modulus five and base
four, samples, and checks `x == 4**power % 5`. Native qfunc expresses the same
computation without gate design, physical wires or bitstring decoding.

```sh
python -m examples.synthesis.functions.arithmetic.modular_exp.modular_exp_example.main
python -m examples.synthesis.functions.arithmetic.modular_exp.modular_exp_example.comparisons.circuit
```

`modular_exp(N,A,x,power)` preserves the source argument order. It maps valid
residues to `A**power*x mod N` and leaves padding values `x>=N` unchanged. The
compiler owns controlled multiplication, scratch storage and cleanup. Both
native and source programs use 7 total qubits, 18 operations and depth 13;
the one workspace bit returns to zero.

The workflow executes locally and draws 1000 seeded samples from executed named
probabilities. Plain tuples replace dataframe rows, with the modular relation
checked for every exact support point and every sampled outcome. It does not
pretend that a cloud sampler or hardware backend ran.

An independent Circuit applies a dense permutation computed with ordinary
classical integer arithmetic. It verifies the unchanged original and the native
program in every raw complex amplitude. Generic tests cover all small basis
columns, padding values, multiple coprime bases, complex coherent inputs,
absolute phase, both objectives, helper composition and existing width/storage/
resource rejections. Reviewed Notebook code-cell hashes are in provenance.
