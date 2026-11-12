# Quantum-program scope: a 67-variable phase oracle

`main.py` retains every variable and every one of the 67 original clauses. Write
the Boolean condition and the phase intent; synthesis constructs clause flags,
the conjunction and the cleanup circuit. This is the complete quantum-program
counterpart. The associated Notebook workflow remains pending.

```python
from math import pi
from flagquantum import synth
from flagquantum.synth import (
    Output, QArray, QBit, allocate, control, hadamard_transform, phase, qfunc,
)

@qfunc
def coherent_oracle(q: Output[QArray[QBit]]) -> None:
    allocate(4, q)
    hadamard_transform(q)
    control(
        (q[0] or not q[1]) and (q[2] or q[1] or q[3]),
        lambda: phase(1, pi),
    )

program = synth.synthesize(coherent_oracle)
```

The original initializes all 67 problem bits to zero. Its exact result is the
same bit word with absolute phase -1, not a search distribution. The native
builder uses 150 physical qubits, 83 clean work qubits, 1033 emitted operations,
depth 629 and 4800 bytes of rank-one MPS state storage. That storage figure is
not process peak memory. Runtime proves the product-state profile before
execution; no dense statevector or approximate truncation is used.

The independent Circuit comparison evaluates the clauses as classical truth on
that original input and checks all 150 restored bits and the absolute sign.
Tests also exercise false and nonzero assignments, both source frontends,
complete complex amplitudes for a four-bit coherent oracle, tautologies,
duplicate literals, read validity and actual resource-budget refusals. This does
not claim exhaustive testing of all 2**67 input assignments.

The bounded extension admits at most eighty clauses, each containing at most
three signed indexed bits, for a phase of pi or -pi. It shares existing generic
control-X lowering and its resource checks. Other phase predicates retain their
existing support limits. Python Boolean syntax is captured inside `@qfunc`;
the host function is never executed to decide a quantum condition.
