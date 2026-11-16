# Set Partition quantum program

The cost is the negative squared residual of the retained weighted sum from67. Every one of the ten weights is retained.

All 3 layers and all 6 real parameters are retained. Parameters contain
all gamma values followed by all beta values. Each layer applies the cost phase
then RX(beta) on each of the 10 bits; there is no factor-of-two change in the
mixer convention.

The source has one bit-array field, `v.x`. The native program declares that
same storage directly as `Output[QArray[QBit]]` named `v`, with unchanged bit
order. This explicit representation is not general native qstruct support.
The test-only source comparison keeps the source field name. User examples use
no source text or importer.

```python
import flagquantum as fq
from examples.synthesis.applications.optimization.set_partition.main import build_program

program = build_program()
result = fq.run(program)
print(dict(program.probabilities(result, registers=("v",))))
print(program.report["resources"])
```

Two nonzero parameter vectors and both width/depth objectives are checked against
independent classical cost tables and explicit Circuit spin-parity rotations.
The Walsh reference uses O(2**10) classical memory and two-qubit parity ladders,
rather than a dense 2**10-square gate matrix. Raw complex128 statevectors,
absolute global phase and zero workspace are checked. The source physical
resource baseline remains separately frozen.

This completes the retained quantum program only. Classical optimizer training,
parameter search and the complete related Notebook workflow remain pending.
