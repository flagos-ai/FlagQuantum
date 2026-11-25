# Native QSVT with a shifted packed word

```sh
python -m examples.synthesis.tutorials.technology_demonstrations.classiq_paper.qsvt.main
python -m examples.synthesis.tutorials.technology_demonstrations.classiq_paper.qsvt.comparisons.circuit
```

The native entry retains all six data bits, three block bits, the signal
auxiliary and four phases. The source's local packed word becomes the explicit
`data_and_deletion` output: its six low bits are data and its top bit is the
block deletion bit. `select`, `projection` and `aux` remain named outputs.
No original output degree of freedom is discarded. This is the quantum program;
the complete Notebook workflow remains separately pending.

```python
import flagquantum as fq
from examples.synthesis.tutorials.technology_demonstrations.classiq_paper.qsvt.main import build_program

program = build_program()
result = fq.run(program)
print(dict(program.probabilities(result, registers=("select", "projection", "aux"))))
print(program.report["resources"])
```

The block encoding uses modular increments and a zero-state reflection. The
root within/apply region computes the deletion marker, applies the exact
projected RZ as two phase polynomials, and uncomputes the marker. This retains
the constant phase and preserves interference. It uses existing ownership and
predicate rules; no compiler cap is widened. Mutable helpers contain only
borrowed unitary effects.

The independent Circuit constructs the modular permutation and reflection
matrices and analytic block-zero phase diagonals. It does not call native
increment or phase-polynomial lowering. Tests compare all complex amplitudes,
source/native axis permutations and every work amplitude, including two nonzero
phase variants and a coherent complex-state inverse test of the borrowed block.

Native resources are 15 physical qubits including five clean work bits, 17,663
operations and depth 14,619. The original has 14 physical qubits including four
work bits, 2,798 operations and depth 2,270. The conservative native reflection
and predicate expansions have a substantial cost gap, recorded without a
performance improvement claim. General optimization is separate from semantic
migration; raw original resource baselines remain intact.
