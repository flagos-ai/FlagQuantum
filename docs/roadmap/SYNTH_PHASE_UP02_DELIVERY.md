# UP-02 native phase capture

Revision 9 UP-02 starts by bringing the existing one-bit pi phase contract to
the native program capture path. A private Python marker lets typed qfunc
bodies borrow initialized registers and capture an unsigned one-bit arithmetic
expression. The existing register verifier, expansion, phase lowering and
exact target checks handle the resulting qreg.phase operation.

The first slice accepts the exact Python float value of pi from a literal or
a static binding. It shares the current unsigned expression capture with
fresh arithmetic assignment. Static nested and repeated calls are covered
through the same register call graph as Qmod.

The test compares every complex amplitude against an independent sign oracle
for a two-qubit nested and repeated program. Zero amplitudes outside the
logical support also check workspace cleanup. Both native capture and Qmod
execute through the existing public synthesis and Runtime workflow. The
marker is internal while the complete phase contract is being developed.

## Public weighted phase

`phase` is exported from `flagquantum.synth`. A real coefficient weights an
unsigned arithmetic expression. For a real constant expression, the operation
applies one global phase. Coefficients and scaled angles use finite binary64.
The exact one-bit pi case retains the predicate path. The Qmod predicate
and explicit RZ/PHASE gates keep their existing contract.

```python
import flagquantum as fq
from flagquantum import synth
from flagquantum.synth import Const, Output, QNum, allocate, hadamard_transform, phase, qfunc


@qfunc
def cost(x: Const[QNum[2]], y: Const[QNum[2]]) -> None:
    phase(x * x + y, 0.8)


@qfunc
def main(x: Output[QNum[2]], y: Output[QNum[2]]) -> None:
    allocate(x)
    allocate(y)
    hadamard_transform(x)
    hadamard_transform(y)
    cost(x, y)
    phase(y, -0.3)
    phase(-0.5, 1.9)


program = synth.synthesize(main, max_qubits=24)
result = fq.run(program)
print(dict(program.probabilities(result, registers=("x", "y"))))
```

The independent amplitude for each logical output is
`exp(i * (0.8 * (x*x+y) - 0.3*y - 0.95)) / 4`. The implementation computes
the unsigned value into clean scratch, applies binary-weighted phase gates,
and reverses the exact arithmetic. A constant global phase uses the existing
X and PHASE gates as an identity times a scalar. The tests check all sixteen
complex amplitudes and zero scratch, both selection objectives, and a Qmod RZ
and hand Circuit reference without removing global phase.
