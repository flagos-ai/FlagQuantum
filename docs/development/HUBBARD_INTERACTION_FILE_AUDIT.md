# Hubbard interaction file audit

Q-0440 and Q-0441 cover the on-site interaction declarations and original tests
from Qualtran v0.7.0 at `bcb17bb25360ed68ce3bc084645db3e929799389`.
The frozen sources matched their Git blobs. The unchanged source test file
returned six passes and six skips.

Both source variants declare one opaque through register named `system`
with shape `(2,)` and word width equal to the lattice side. The native
source signature matches the name, through direction, shape, ordered
indices and width for concrete and symbolic lengths. The concrete source
resource model derives its port width from this signature. The native
execution binds two row-major spin registers of length `length**2`.

The ordinary declaration repeats `Rz(angle * hubb_u)` once per site.
The HWP variant declares two `HammingWeightPhasing` calls of width
`length**2 // 2` and exponent `angle * hubb_u`. Source tests compare
these child parameters, synthesis precision, multiplicities, gate
categories and logical peaks. Symbolic signature support is independent
of concrete cost evaluation.

## Native interaction

The standalone operation reuses the on-site XOR basis and diagonal
emitter in the Suzuki step. It applies
`exp(i * timestep * hubb_u * sum(ZZ) / 4)` using `Rz(-timestep * hubb_u / 2)`
on each parity wire. The adjoint reverses the angle. HWP uses two equal
batches, preserves the scalar phase and reuses the same work and
feedback slots across batches. When the number of sites is odd, an
additional direct rotation covers the last parity wire, which is
omitted by the frozen two-batch cost formula. The one-site circuit
uses the ordinary direct rotation. A two-batch resource declaration
requires a positive batch width, so the one-site HWP binding is rejected.

For the two-by-two example, the ordinary native circuit emits eight CX
and four Rz with no work or measurements. The HWP variant emits eight
T-family instructions and two measurements, reusing three work wires.
The independent HWP source model gives a peak of five logical qubits.
Tests compare the ordinary full complex matrix in both directions
against the shifted-ZZ diagonal, and all four HWP measurement histories
against that diagonal on a coherent state. The full output vectors show
the work register restored to zero. A three-by-three sparse phase test
checks that the odd site's phase is present in both modes.

Q-0441 closes with the original checks and these native comparisons.
Q-0440 retains the locked decomposition and synthesis boundaries.
Both original variants raise `DecomposeNotImplementedError`, and
the source model's angle parameter and width differ from the native
primitive angle and bound width. The exact native rotations use no
synthesis tolerance.

```python
from flagquantum.algorithms.primitives.hubbard_lattice import HubbardInteraction
from flagquantum.dynamic import DynamicCircuit

op = HubbardInteraction.make_on(
    length=2,
    up=(0, 1, 2, 3),
    down=(4, 5, 6, 7),
    timestep=0.23,
    hubb_u=-1.7,
    hamming_weight=True,
    workspace=(8, 9, 10),
)
ir = DynamicCircuit(11).append(op).to_ir()
assert sum(gate.name == "measure" for gate in ir.instructions) == 2
```
