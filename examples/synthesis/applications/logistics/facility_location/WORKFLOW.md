# Facility location with shared Boolean predicates

The native program uses application decision `assignments[4 * facility + customer]`
and keeps all twelve original delivery costs, once-per-customer squared penalties,
the squared count of empty facilities, and all five QAOA cost/mixer layers.

```python
import flagquantum as fq
from examples.synthesis.applications.logistics.facility_location.main import build_program

program = build_program(tuple((-1) ** i * (0.004 + 0.001 * i) for i in range(10)))
result = fq.run(program)
print(program.report["resources"])
print("State norm:", result.statevector().norm().item())
```

The first five angles are cost parameters; five RX mixer angles follow. The cost
helper writes the count of empty rows as products of Boolean complements. The
compiler preserves and shares these predicates, then uncomputes them. No user
work register or physical wire assignment is required.

Two nonzero parameter sets, both optimization objectives and all 4,096 classical
assignments are checked. An independent Circuit expands bounded truth factors
into spin parities; it does not invoke synthesis or load external source text.
Original field-row readout order is checked separately. Complete complex
amplitudes must match within 1e-12, including global phase; hidden work probability
must remain below 1e-24.

Observed default resources are seventeen qubits, five work bits, 423 operations
and depth 171. The immutable original uses 444 operations and depth 177 with the
same width. These observations cover this retained model, not arbitrary resource
optimality. The shared Boolean candidate keeps its existing word limits; higher
physical masks can require conservative lowering and must pass actual memory
budgets. The full associated Notebook workflow remains pending.
