# Native commuting-Hamiltonian HHL

Run from the repository root:

```sh
python -m examples.synthesis.algorithms.quantum_linear_solvers.hhl.main
python -m examples.synthesis.algorithms.quantum_linear_solvers.hhl.comparisons.circuit
```

The native program prepares a two-bit RHS, estimates the commuting Hamiltonian's
eigenphase into a signed four-bit all-fractional counter, applies the retained
approximate reciprocal rotation and uncomputes phase estimation. Users declare
the model and phase precision; allocation, quantum composition and lowering use
existing native quantum functions. No physical wire numbers appear in `main.py`.

```python
import flagquantum as fq
from examples.synthesis.algorithms.quantum_linear_solvers.hhl.main import build_program

program = build_program()
result = fq.run(program)
print(dict(program.numeric_probabilities(result, registers=("solution", "phase_var", "indicator"))))
print(program.report["resources"])
```

The phase counter decodes as a signed word divided by 16. Rotation-table lookup
uses the stored four-bit word, preserving that numeric declaration. `data.py`
retains all four Pauli coefficients and the 16 approximate RY/CX pairs. Moving
the even-parity X network through Y rotations gives the raw-word angle table;
it does not replace the approximate source network by ideal reciprocal values.
The source's skip-control wrappers have no effect at this uncontrolled root.
This example does not introduce generic skip-control or outer-control semantics.

Native QPE repeats the first-order evolution for each power. The source scales
the evolution time. Since the terms commute, both preserve the same absolute
unitary, including the identity term's phase under control. Native execution
uses seven physical qubits, no work, 581 operations and depth 435. The original
uses seven qubits, no work, 207 operations and depth 150. Repeated evolution
costs more than time-scaled evolution; this migration claims semantic acceptance,
not equal resource efficiency.

The independent Circuit uses the analytic Hamiltonian matrix, its matrix
exponentials, Fourier matrices and the original 16-pair inversion network.
Acceptance covers every complex amplitude of the original and two additional
normalized nonzero real RHS vectors. Separate coherent-selector tests exercise
the inversion on a nonzero complex indicator, including every signed raw word;
checked inverse and budget refusals are also covered. Numeric probabilities
perform decoding without manual bit strings. The signal branch is not silently
postselected and the complete Notebook workflow remains separately pending.
