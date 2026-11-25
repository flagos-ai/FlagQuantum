# Native Shor period finding

Run from the repository root:

```sh
python -m examples.synthesis.algorithms.number_theory_and_cryptography.shor.main
python -m examples.synthesis.algorithms.number_theory_and_cryptography.shor.comparisons.circuit
```

The native program prepares a ten-bit phase counter, applies modular
exponentiation with modulus21 and base11 to a five-bit residue initialized at1,
then applies the inverse Fourier transform. This preserves the original quantum
period-finding stage. Classical factor recovery and the associated Notebook
workflow require separate acceptance.

```python
import flagquantum as fq
from examples.synthesis.algorithms.number_theory_and_cryptography.shor.main import build_program

program = build_program()
result = fq.run(program)
print(dict(program.numeric_probabilities(result, registers=("phase_var",))))
print(program.report["resources"])
```

The counter stores ten raw bits. Modular exponentiation uses their integer bit
pattern as the exponent; the declared all-fractional output decodes each measured
word as `word/1024`. Named results perform this decoding automatically. The
private residue is dropped from the public output, but remains a dirty quantum
environment and must be included when checking complete statevectors.

The default native circuit uses18physical qubits, including10phase bits,5retained
residue bits and3clean work bits. It has2,116operations and depth1,849. The original
path has2,108operations and depth1,844 with the same qubits. Native initialization
increments an allocated zero word to1; it is not claimed to match the optimized
literal-XOR preparation cost.

The independent Circuit constructs each controlled modular permutation directly
and leaves invalid residue encodings unchanged. Its inverse DFT is an analytic
matrix. It does not call compiler modular-cycle decomposition or QFT lowering.
Tests compare all complex amplitudes with the unchanged original, check every
work amplitude, exercise two additional coprime parameter pairs, verify phase
bit order and numeric decoding, and reject invalid parameters and budgets.
