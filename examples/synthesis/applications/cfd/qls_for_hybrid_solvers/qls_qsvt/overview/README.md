# Nozzle inverse with quantum singular value transformation

Run the [notebook](walkthrough.ipynb) or its local CPU workflow from the repository
root.

```sh
python -m examples.synthesis.applications.cfd.qls_for_hybrid_solvers.qls_qsvt.overview.main
```

The workflow uses both original matrices and right-hand sides, retains the
24-to-20 Pauli trim for the eight-row example, and uses the original banded
encoding for the sixteen-row example. The inverse polynomial has degree 101
or 255, respectively. Its cutoff uses the smallest singular value expressed in
the block encoding's normalized coordinates.

The example fits a fresh odd symmetric phase sequence for each polynomial.
The bounded Newton solve uses the analytic Jacobian and checks the response on
4,097 points. The phase fitting follows the symmetric-QSP formulation of Dong,
Lin, Ni and Wang. [Paper](https://arxiv.org/abs/2307.12468)

The QSVT circuit synthesizes the original right-hand side, then applies the two
projector-phase sequences needed to recover the real response. The CPU example
packs each alternating block/adjoint sequence as an exact matrix inside a
controlled simulator instruction. Reported resource counts refer to these
represented instructions. The independent sequential reference checks the
complete state, and the SVD reference checks the zero-block inverse action
before comparison with the dense solution.

The [native block composition](native.py) also exposes the sixteen-row matrix as
bounded `select_rotation` tables, signed cyclic shifts and within/apply. It retains
all 45 matrix entries and six nonzero diagonals (-8, -7, -1, 0, 7, 8). The original
right-hand side uses eleven total qubits, three clean work bits, 566 logical
instructions and depth 527. Complete column tests cover every selector and flag
branch, including unused selector values, rather than only the success subspace.

```python
import flagquantum as fq
from examples.synthesis.applications.cfd.qls_for_hybrid_solvers.chebyshev_approximation.overview.model import source_matrix
from examples.synthesis.applications.cfd.qls_for_hybrid_solvers.qls_chebyshev_lcu.overview.workflow import input_vector
from examples.synthesis.applications.cfd.qls_for_hybrid_solvers.qls_qsvt.overview.native import banded_program

rhs = input_vector("nozzle_008_b", 4)
program, scale = banded_program(source_matrix(), rhs / rhs.norm())
result = fq.run(program)
print(program.report["resources"])
print(program.probabilities(result, registers=("selection", "flag"))[(0, 0)])
```

This native block is a prerequisite for a native inverse. The full native
102-phase and 256-phase sequences currently refuse at the 4096-effect frontend
limit. The packed Circuit solve above remains the accepted full inverse; neither
a shorter phase sequence nor a matrix-packed replacement is counted as native
QSVT completion. The pending effect-capacity proposal records the separate
admission decision.
