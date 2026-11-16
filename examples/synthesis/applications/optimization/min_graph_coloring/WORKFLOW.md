# Minimum graph coloring quantum program

The native program names each decision `colors[5 * color + vertex]`. It prepares
all fifteen decisions in superposition and runs all six original cost/RX layers.

```python
import flagquantum as fq
from examples.synthesis.applications.optimization.min_graph_coloring.main import build_program

params = tuple((-1) ** i * (0.004 + 0.001 * i) for i in range(12))
program = build_program(params)
result = fq.run(program)
print(program.report["resources"])
print("State norm:", result.statevector().norm().item())
```

The first six parameters are cost angles; the last six are RX mixer angles.
The cost helper retains the empty-color rewards, squared once-per-vertex
constraints, constant -3 and the source's squared conflict count. Each of the
five undirected edges appears twice before squaring in the original; the native
expression writes the equivalent coefficient -40 on the squared undirected sum.
It does not substitute a conventional graph-coloring objective.

Independent reference Circuits use bounded classical truth factors, transformed
into spin parities without synthesis or an importer. Tests check all 32,768
classical assignments against original arithmetic and the independent constraint
model. Two nonzero parameter sets and both optimization objectives compare full
complex amplitudes within 1e-12, including absolute phase and source row order.
Hidden workspace probability must be below 1e-24.

Native default resources are fifteen qubits, zero work bits, 19,975 operations
and depth 13,069. The frozen original has 17,881 operations and depth 13,284.
These are measured compilation choices, not a resource-optimality claim.
Qubit, operation, depth and memory budgets are checked before execution.
The associated complete Notebook workflow remains pending.
