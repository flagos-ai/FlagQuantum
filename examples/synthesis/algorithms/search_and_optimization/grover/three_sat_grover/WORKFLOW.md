# Native 3-SAT phase search

The mark helper represents each original SAT clause as a Boolean complement
product. Its phase flips exactly satisfying assignments 8 and 12. A diffuse
helper reflects about the uniform state; main repeats both helpers r times.

```python
import flagquantum as fq
from examples.synthesis.algorithms.search_and_optimization.grover.three_sat_grover.main import build_program

program = build_program(2)
result = fq.run(program)
print(dict(program.probabilities(result, registers=("q",))))
print(program.report["resources"])
```

The diffuser includes a constant pi phase as well as the zero-state phase,
so it implements 2|s><s|-I with the original absolute sign. This is a concrete
oracle specialization, not a new generic Grover or quantum-callback API.
The builder admits integer iteration counts from zero through 32.

For r=0/1/2 and both objectives, original and native programs match independent
classical-oracle and dense reflection Circuits in all complex amplitudes within
1e-12, including global phase, bit order and hidden-work probability below 1e-24.
Invalid iteration bindings and actual qubit, operation, depth and memory limits
are checked. The native Python directory uses a valid identifier; source identity
and frozen source bytes still retain the original 3_sat_grover name.

At r=2 native resources are four qubits, zero work bits, 172 operations and
depth 149. The original uses five qubits, one clean work bit, 46 operations and
depth 33. This width/gate difference is retained as measured evidence rather than
an equal-resource claim. The complete associated Notebook workflow remains
pending.
