# UP-03 native scoped cleanup

The first UP-03 slice captures adjacent Python `with within()` and
`with apply()` blocks using private compiler markers. A preparation block
declares sized unsigned locals with annotated arithmetic assignments.
The apply block admits phase operations and static Const-only qfunc calls.
Each local is uncomputed and released in reverse order using the existing
register effect verifier and arithmetic adjoint.

```python
from flagquantum.compiler._construction.program_types import apply, within
from flagquantum.synth import Const, QNum, phase, qfunc


@qfunc
def cost(x: Const[QNum[2]], y: Const[QNum[2]]) -> None:
    with within():
        score: QNum[4] = x * x + y
    with apply():
        phase(score, 0.8)
```

A native program composes this block through ordinary qfunc calls. The
equivalent one-bit Qmod within/apply path and the native path are checked
against an independent complete-state sign oracle. The two-bit weighted
score case verifies all sixteen complex amplitudes and every cleanup qubit.
Tests also reject a dependency mutation inside apply and a local read after
its release.

Public marker admission and general checked inversion are subsequent
UP-03 work. Nested static qfunc calls are covered here. Nested lexical
within/apply blocks require their own admission and test cases.
