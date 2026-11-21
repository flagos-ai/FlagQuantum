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

The executed program composes the full source inverse in native qfunc: real
amplitude preparation, selected Pauli or six-band blocks, their checked adjoints,
and shared compute/use/uncompute projector predicates. Both branch streams retain
all 102 or 256 fitted phases. Classical helpers derive tables and fit phases;
users do not allocate work bits or supply physical wire numbers. The independent
sequential reference compares the entire complex state without phase alignment;
the SVD reference checks the zero-block inverse action and original solution error.
The separate matrix-packed Circuit builder is reference-only.

| Source inverse | Total qubits | Clean work bits | Logical instructions | Depth |
| --- | --- | --- | --- | --- |
| Eight rows, degree 101, 102 phases | 13 | 4 | 40447 | 35575 |
| Sixteen rows, degree 255, 256 phases | 12 | 3 | 87012 | 81109 |

These are bounded local CPU instruction counts, not a hardware cost or speedup
claim. Both complete states pass at absolute tolerance `1e-10`; work probability
is below `1e-24`. No shortened phases or dense response matrix enter the native
execution. Construction retains the 100000-operation ceiling.

The [native source block](native.py) retains all 45 matrix entries and six
nonzero diagonals (-8, -7, -1, 0, 7, 8). Generic signed-place modular shifts reduce
its original-RHS cost from 566 instructions/depth 527 to 343/depth 307, still
with eleven total qubits and three work bits. Complete column tests cover every
selector and flag branch, including unused selector values.

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

For the full original solves, run `solve(False)` and `solve(True)` from
`workflow.py`, or run every Notebook cell. The shown qfunc definitions are the
same composition used by those solves. Both `full_state_error` and
`workspace_probability` are reported separately from postselection probability
and solution error. General symbolic QSVT APIs remain a separate design scope.
