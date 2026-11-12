# Native ordered Pauli evolution

The user authorized native high-level programs across all five example categories.
This unreleased addition exposes `suzuki_trotter(operator, time, order,
repetitions, target)` in the established positional order. Classification is
`single_device_fast_path`; `release_authorized=false`. Stable Core and frozen
source/resource baselines remain unchanged.

`operator` is the existing immutable `fq.Observable`, bound outside the qfunc.
Capture snapshots 1..256 real Pauli terms without executing constructors, hooks
or host expressions. Observable indices address logical target elements, with
element zero the least significant bit. This explicit binding differs from
physical wire labels on a Circuit. Complete initialized writable QNum/QArray
registers and complete mutable-helper views are admitted; partial views,
quantum/classical shadowing, dynamic operator construction and new inverse or
controlled forms are refused. Const ownership remains enforced by the shared
register verifier.

Time uses existing bounded finite classical-real binding. Coefficients must be
built-in finite real numbers within +/-2**53. Orders 1 and 2 and 1..256 static
repetitions reuse the shared compiler product-formula schedule. Order 1 visits
terms forward with time/repetitions; order 2 visits forward then backward with
half that time. Each step emits existing `qreg.pauli_rotation`, representing
exp(-i * coefficient * step * Pauli), including identity global phase. Terms
are never silently sorted or combined. This is the declared product formula,
not an exact exponential of a general noncommuting sum or an error guarantee.
The 4096-effect capture bound and existing operation/qubit/memory budgets apply.
For otherwise over-budget consecutive calls on a complete one-to-three-qubit
register, capture shares the existing Qmod compact product transformation.
A buffer holds at most 256 calls; composition emits a checked `qreg.unitary` for
each 64-call chunk. Calls retain all terms, declared order and repetitions.
A phase-equivariant Newton polar correction is admitted only for a finite Gram
residual at most 1e-11. Small programs retain their previous rotation expansion.
Intervening effects, changed targets and helper ownership boundaries flush the
buffer; Const, initialization and alias verification remain mandatory. Physical
operation, depth and memory budgets still apply after unitary lowering.
No Hamiltonian type, numerical kernel or execution route is added.

```python
import flagquantum as fq
from flagquantum.synth import (
    Output, QArray, QBit, allocate, qfunc, suzuki_trotter, synthesize,
)

HAMILTONIAN = 0.3 * (fq.Z(0) @ fq.Z(1)) + 0.7 * fq.X(0) + 0.2 * fq.X(1)


@qfunc
def main(time: float, q: Output[QArray[QBit]]) -> None:
    allocate(2, q)
    suzuki_trotter(HAMILTONIAN, time, 1, 10, q)


program = synthesize(main, classical_inputs={"time": 1.0})
print(dict(program.probabilities(fq.run(program), registers=("q",))))
```

Acceptance compares independent dense Circuit products, raw complex amplitudes,
both objectives, positive/negative nonzero times, term/bit order, coherent inputs,
identity global phase, helper composition and zero workspace. Invalid or mutable
operator data, aliases, nonfinite angles, unsupported views/orders/repetitions
and exhausted resource budgets must fail before execution. Native quantum
program coverage and complete Notebook workflows remain separately counted.
