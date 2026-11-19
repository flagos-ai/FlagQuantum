# QMCI Pi estimation

Run the complete native workflow from the repository root:

```bash
python -m examples.synthesis.tutorials.basics.qmci_pi_estimation.overview.main
```

Open [walkthrough.ipynb](walkthrough.ipynb) for all thirteen source code cells,
the four-bit lattice plot, classical grids, three-bit QPE, iterative estimates
and final error bars. The original quantum-only example in the parent directory
is unchanged. This workflow retains three bits per coordinate for QPE and all
four iterative coordinate precisions, instead of silently shrinking a scenario.

A bound synthesis-time integer determines storage without designing gates:

```python
import flagquantum as fq
from flagquantum.synth import Output, QNum, allocate, hadamard_transform, qfunc, synthesize

@qfunc
def uniform(bits: int, q: Output[QNum]) -> None:
    allocate(bits, q)
    hadamard_transform(q)

program = synthesize(uniform, classical_inputs={"bits": 3}, max_workspace=0)
print(dict(program.probabilities(fq.run(program), registers=("q",))))
```

The eight integer labels each have probability 1/8. Widths specialize separately
for each synthesis; quantum values and arbitrary Python calls cannot set sizes.
Numeric type annotations still require literal metadata.

[operations.py](operations.py) expresses uniform coordinates, the strict
`x*x+y*y < 2**(2*bits)` marker and Grover composition. Users do not supply gates,
physical wire numbers or bitstring decoding. [workflow.py](workflow.py) keeps
independent dense Circuit checks separate from native construction and derives
observation counts from actual local state probabilities.

The original QPE reports `4*sin(pi*phase)`. We preserve that result and label the
`4*sin(pi*phase)**2` probability comparison separately. Neither coarse phase
resolution nor finite lattice bias is a confidence interval for continuum pi.
The bounded count-only iterative estimator is reused from the existing quantum
counting example. It uses union-bound Hoeffding intervals and checked monotone
quadrant inversion; it is not implementation-equivalent to hosted Classiq IQAE.
Local seeds and cached state probabilities model local observations, not QPU
execution time or a quantum speedup. Actual shot totals are printed separately
from the lattice-size horizontal axis. Dropped coordinates remain an entangled
environment; only genuine working storage must return to zero.

See [WORKFLOW.md](WORKFLOW.md) and [provenance.json](provenance.json) for source
mapping, accepted scope and limitations.
