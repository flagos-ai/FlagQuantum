# Fourier transform workflow

The source Notebook has two code cells: allocate a four-qubit register and
apply QFT inside qfunc, then synthesize it. The native `main.py` preserves this
complete computational workflow. It additionally executes locally and prints
sixteen outcomes, each with probability 1/16.

```sh
python -m examples.synthesis.functions.qft.main
python -m examples.synthesis.functions.qft.comparisons.circuit
```

The Circuit comparison constructs the Fourier matrix from its mathematical
definition, independently of the shared compiler kernel. Acceptance tests check
all columns for widths 1..4 and complex coherent inputs, including offset wires,
absolute phase and zero workspace. Uniform probabilities alone are insufficient.
The native/original four-bit resources agree: 4 qubits, 12 operations, depth 8.

Complete initialized writable registers and whole-register mutable helpers are
supported. Partial views, uninitialized or Const storage are refused. This batch
does not extend native controlled or inverse helper admission. Source identities
and reviewed Notebook code-cell hashes are recorded in `provenance.json`.
