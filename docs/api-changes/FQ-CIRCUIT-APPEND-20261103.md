# Append bound operations to a circuit

Status: proposed for API-owner review.

A user building a SELECT circuit can add a library operation in the same order
as its surrounding gates. The circuit owns the mutation, so this entry point
belongs on `Circuit`.

```python
import flagquantum as fq
from flagquantum.algorithms.primitives.multiplexers import (
    SelectedMajoranaFermion,
    SelectPauliLCU,
)

circuit = fq.Circuit(3)
majorana = SelectedMajoranaFermion.make_on(selection=[0], target=[1, 2])
select = SelectPauliLCU(1, 2, ("XI", "IZ"))
circuit.append(majorana)
circuit.append(select.make_on(selection=[0], target=[1, 2]))
```

`append(op)` accepts an operation with bound qubit indices and an
`append_to(circuit)` method, and returns the receiving circuit. These SELECT
operations each emit one structured instruction. Serialization preserves the
instruction without a matrix. Compilation or execution lowers it through the
existing operator schema. `compose` continues to place a circuit or IR program
on chosen qubits. Existing `append_to` callers keep their current behavior.

Release note: `Circuit.append(op)` accepts bound SELECT operations.
