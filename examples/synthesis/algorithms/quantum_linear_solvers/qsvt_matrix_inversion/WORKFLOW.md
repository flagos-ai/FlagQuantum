# Native singular-value matrix inversion

Run from the repository root:

```sh
python -m examples.synthesis.algorithms.quantum_linear_solvers.qsvt_matrix_inversion.main
python -m examples.synthesis.algorithms.quantum_linear_solvers.qsvt_matrix_inversion.comparisons.circuit
```

The program retains the complete eight-by-eight block encoding, four input
amplitudes and all 70 inversion phases in `data.py`. They are immutable numerical
model data; execution never reads a Qmod source file. This is the retained quantum
program, not a general phase-sequence generator or the complete Notebook workflow.

```python
import flagquantum as fq
from examples.synthesis.algorithms.quantum_linear_solvers.qsvt_matrix_inversion.main import build_program

program = build_program()
result = fq.run(program)
print(dict(program.probabilities(result, registers=("state", "block", "aux"))))
print(program.report["resources"])
```

The original structured output is exposed as a two-bit `state` array and a
separate `block` bit. The signal auxiliary `aux` is also public. The block is
the most significant matrix bit: `unitary(MATRIX, (state, block))` declares the
storage sequence in increasing significance. The Circuit counterpart uses
explicit wire order `(block, state[1], state[0])` in matrix significance, which
is physical wires `(2, 0, 1)` in this program. Original-source state comparisons
permute its packed `(block, state, aux)` axes into native `(state, block, aux)`.

The inversion starts with a projected rotation, alternates inverse and forward
block encodings, and ends with the final rotation. All phases, including the
large final angle, are retained. The independent Circuit constructs a four-bit
phase diagonal analytically; it does not replay the controlled-X implementation.
Tests compare absolute amplitudes, not just probabilities, for the original
sequence and two nonzero perturbations. They verify the postselected scaled
linear solution for the original sequence and inverse restoration of all state
and work amplitudes. No postselection or renormalization is applied to the full
state comparison.

Current native lowering uses five physical qubits, including one clean work bit,
68,023 operations and depth 59,977. The original uses four physical qubits,
67,915 operations and depth 59,872. The extra work and operations are reported
honestly; this migration does not claim an optimized resource improvement.

The checked inverse is compiled separately and executed on the independent full
forward state. A single forward-plus-inverse compilation would exceed the
current 100,000-operation construction cap; a larger user resource constraint
does not enlarge that compiler safety cap. This test preserves the cap and
checks composition at the existing Circuit execution boundary.
