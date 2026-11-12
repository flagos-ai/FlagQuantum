# Native thresholded MaxCut phase search

The mark helper computes the original six-edge cut size and marks cut_size<=4.
It intentionally retains this original condition, which differs from the >=4
threshold in another frozen Grover example. Main prepares five decisions in
superposition and repeats mark/reflection helpers r times.

```python
import flagquantum as fq
from examples.synthesis.algorithms.search_and_optimization.grover.grover_max_cut.main import build_program

program = build_program(2)
result = fq.run(program)
print(dict(program.probabilities(result, registers=("q",))))
print(program.report["resources"])
```

The reflection preserves the absolute sign 2|s><s|-I by including the constant
pi phase. No generic Grover callback API is introduced. The example builder
admits integer iteration counts from zero through 32.

Independent classical cut counts check all 32 assignments, including cut sizes
four and five. For r=0/1/2 and both objectives, original and native complete
complex amplitudes match independent oracle/reflection Circuits within 1e-12,
including global phase, bit order and hidden-work probability below 1e-24.
Invalid bindings and actual resource ceilings refuse before execution.

Native r=2 resources are five qubits, zero work bits, 749 operations and depth 652;
the original uses seven qubits, two clean work bits, 148 operations and depth 122.
Default objectives and an explicit zero-workspace ceiling are measured separately.
The native diagonal expansion currently trades fewer work bits for more gates;
this semantic counterpart makes no resource-optimality claim. The complete
associated Notebook workflow remains pending.
