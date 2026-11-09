# Native arithmetic sampling workflow

The source Notebook defines the unsigned quantum expression, synthesizes it,
and samples its outputs. The native main executes the same computation and
samples 1000 times from actual executed named probabilities with fixed seed 17.
Expected counts: `{(4, 7, 3): 1000}`.

```sh
python -m examples.synthesis.functions.arithmetic.modulo.modulo_example.main
```

Local sampling replaces the cloud call; this is not hardware execution. Original
source bytes remain test-side. Provenance records the reviewed Notebook code-cell
hashes. The modulo example preserves five-bit a/b input domains through standard
`Annotated[QNum, 5]` metadata and fresh initialization equivalent to XOR on zero.
The shared Compiler verifies exact power-of-two remainder, ownership and budgets.
Independent Circuit tests compare raw complex states, including coherent inputs,
absolute phase, bit order and zero hidden workspace. General modulo and native
in-place XOR remain outside this admission.
