# UP-03 native scoped cleanup

Native qfunc programs use adjacent Python `with within()` and
`with apply()` blocks to scope local arithmetic. A preparation block
declares sized unsigned locals with annotated arithmetic assignments.
The apply block admits phase operations and static Const-only qfunc calls.
Each local is uncomputed and released in reverse order using the existing
register effect verifier and arithmetic adjoint.

```python
from flagquantum.synth import Const, QNum, apply, phase, qfunc, within


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

## Checked inverse composition

`with adjoint():` reverses static calls to Const-only qfunc definitions. The
admitted inverse covers exact phase operations, including constant global
phase, and clean arithmetic scopes. For a local scope the inverse keeps
the arithmetic preparation and cleanup around the reversed phase body.
The compiler rejects calls with Output effects and validates the fully
expanded inverse with the existing register verifier.

```python
import flagquantum as fq
from flagquantum import synth
from flagquantum.synth import (
    Const, Output, QNum, adjoint, allocate, apply, hadamard_transform, phase, qfunc, within,
)


@qfunc
def cost(x: Const[QNum[2]], y: Const[QNum[2]]) -> None:
    with within():
        score: QNum[4] = x * x + y
    with apply():
        phase(score, 0.8)


@qfunc
def main(x: Output[QNum[2]], y: Output[QNum[2]]) -> None:
    allocate(x)
    allocate(y)
    hadamard_transform(x)
    hadamard_transform(y)
    with adjoint():
        cost(x, y)


program = synth.synthesize(main, max_qubits=24)
result = fq.run(program)
print(dict(program.probabilities(result, registers=("x", "y"))))
```

This example has sixteen equal probabilities and amplitudes
`exp(-0.8j * (x*x+y)) / 4`. The tests check inverse-only and forward/inverse
programs including dependent locals, nested static calls and an absolute
global phase. Arbitrary host callbacks and nested lexical within/apply
blocks are outside the admitted subset.

For the dependent arithmetic expression `(x*y+x)*(x*y+y)`, the local phase
workflow uses 18 qubits and 16 workspace qubits with retain, and 17 qubits
and 15 workspace qubits with recompute. The corresponding logical costs are
217 operations at depth 150 and 229 operations at depth 178. Both produce
the same complete complex amplitudes and clean workspace.
