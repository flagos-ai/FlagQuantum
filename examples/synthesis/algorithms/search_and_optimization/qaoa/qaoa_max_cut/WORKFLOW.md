# Native four-layer MaxCut

A named cost helper represents six graph edges, normalized by six. An ansatz
helper composes four cost/mixer layers and a main function prepares all five
vertex decisions in superposition.

```python
import flagquantum as fq
from examples.synthesis.algorithms.search_and_optimization.qaoa.qaoa_max_cut.main import build_program

program = build_program((-0.17, 0.23, -0.31, 0.41, -0.53, 0.67, -0.71, 0.83))
result = fq.run(program)
print(dict(program.probabilities(result, registers=("v",))))
print(program.report["resources"])
```

The first four angles are gamma; the final four are beta. The original's fixed
cost callback becomes a direct native helper call. No general callable quantum
ports or classical optimizer workflow are introduced.

Independent references count cut edges on all 32 assignments and construct
parity-phase/RX Circuits. Two nonzero parameter sets and both objectives compare
complete complex amplitudes of original and native programs within 1e-12,
including absolute phase, bit order and hidden work probability below 1e-24.
Zero angles, invalid bindings and physical resource ceilings are also checked.

Observed native and original resources are five qubits, zero work bits,
64 operations and depth 25. The complete associated Notebook workflow remains
pending.
