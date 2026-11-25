# Static unitary workflow

The source Notebook has two code cells: define a diagonal complex matrix and
apply it to a fresh two-qubit register inside qfunc, then synthesize. The native
program covers that complete computation, using an immutable matrix tuple.

```sh
python -m examples.synthesis.functions.unitary.main
python -m examples.synthesis.functions.unitary.comparisons.circuit
```

The example additionally executes locally and checks a direct matrix Circuit.
The zero input stays zero; acceptance also checks all columns of nontrivial
complex unitaries and coherent inputs, so the identity action on zero cannot
hide an incorrect transform. Raw complex amplitudes, bit order, absolute phase
and clean workspace are checked.

Arrays, contiguous slices and explicitly listed bits all treat local position
zero as least significant. Matrix data are literals or immutable numeric tuples;
mutating host matrix objects during capture and arbitrary constructors are refused.
No additional numerical kernel, remote execution or hardware claim is introduced.
Notebook code-cell hashes are recorded in provenance.
