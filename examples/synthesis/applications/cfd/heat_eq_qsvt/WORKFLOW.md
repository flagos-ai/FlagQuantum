# Native heat-equation QSVT composition

```sh
python -m examples.synthesis.applications.cfd.heat_eq_qsvt.main
python -m examples.synthesis.applications.cfd.heat_eq_qsvt.comparisons.circuit
```

The source's three data bits and deletion ancilla form the explicit four-bit
`data_and_deletion` word. The deletion bit is the most significant bit. The
two-bit `select`, single `rotation` and signal `aux` are separate native outputs.
The original eight amplitudes initialize the deletion-zero half of the word;
the other eight amplitudes are zero. No data or block degree of freedom is
traced out. This covers the retained quantum program, not the complete Notebook.

```python
import flagquantum as fq
from examples.synthesis.applications.cfd.heat_eq_qsvt.main import build_program

program = build_program()
result = fq.run(program)
print(dict(program.probabilities(result, registers=("select", "rotation", "aux"))))
print(program.report["resources"])
```

The borrowed block composes probability preparation, selected Y rotations,
controlled modular increments and inverse preparation. A root within/apply
region computes and cleans the block-zero predicate and uses two exact phase
polynomials for the projected RZ, including its constant phase.

The original receives four phases, but its loop consumes only the first three.
The native example preserves all four values and the actual three-rotation
behavior. An explicit test changes only the unused fourth phase and verifies
an unchanged full state. This migration does not silently repair upstream
algorithm intent.

Independent Circuit matrices express the state preparation, selector rotations,
modular shifts and projected phase diagonals. Tests compare every complex
amplitude of the original, two nonzero phase variants and two additional
normalized nonzero input vectors, with explicit source/native port mapping.
A complex uniform-state test checks borrowed block/inverse restoration and work
cleanup. Native execution uses 11 physical qubits including three clean work
bits, 210 operations and depth 160. The original uses 11 qubits with three work
bits, 267 operations and depth 202. These are actual static instruction counts,
not a general optimization guarantee.
