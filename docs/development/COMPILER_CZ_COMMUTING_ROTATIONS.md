# Commuting rotations after native CZ synthesis

Variational circuits alternate entangling and local rotation layers. Native
CZ synthesis can expose RZ rotations on opposite sides of a CZ that commute
and can be combined. The existing local Euler cleanup flushes at each CZ,
so it cannot merge these rotations on its own.

The private `_cz_blocks._candidate` now applies the existing
`merge_commuting_rotations` after local Euler cleanup. This reuses Compiler's
structural commutation analysis and exact-zero rotation merge; it does not add
a second rule table, numerical solver, public API or dependency. The original
Qiskit-derived two-qubit synthesis and Cirq-derived local cleanup retain their
attribution. This composition is a FlagQuantum change, not a new upstream port.

## Admission and evidence

Only generated numeric RX/RZ/CZ proposals reach this step. The input block limit
of 64 operations, raw proposal growth limit, target admission and strict CZ
reduction are checked before cleanup, as before. Merging removes operations;
it cannot use a smaller final output to admit an oversized raw proposal.

After merging, the complete 4x4 operator must still agree with the source up to
global phase at residual `1e-11`. Target artifact compilation independently
rechecks the replacement and records all source and output indices through the
existing v4 evidence. Scheduling consumes the resulting program. Symbolic,
trainable, annotated and custom-matrix boundaries retain their existing path;
fixed-pulse targets and final layout contracts are unchanged.

The regression circuit below previously produced 18 native instructions.
It now produces 16, including the same two CZ gates. This is a development
example, not a held-out application result or a cross-SDK ranking. Additional
commutation analysis adds compilation work; fewer emitted rotations do not
establish lower compilation latency, physical depth, error or execution time.

## Reproduce the affected workflow

The existing private pass extension isolates this development example:

```python
import flagquantum as fq
from flagquantum.compiler import compile
from flagquantum.compiler.optimization._cz_blocks import resynthesize_cz_blocks

circuit = fq.Circuit(2)
for x_angle, z_angle in (
    (0.17, 0.23), (-0.31, 0.41), (0.53, -0.61), (-0.71, 0.83)
):
    circuit.cz(0, 1).rx(0, x_angle).rz(1, z_angle)
compiled = compile(
    circuit, optimization_level=0,
    passes={"example.cz": resynthesize_cz_blocks},
)
assert len(compiled.instructions) <= 16
assert sum(gate.name == "cz" for gate in compiled.instructions) == 2
```

This private transform is also used automatically by the existing eligible
target-artifact workflow. Its integration regression checks native output,
scheduling, complete operators, serialized lineage and independent evidence
verification, with and without routing:

```bash
python -m pytest tests/unit/test_compilation_target_cz_resynthesis.py \
  tests/unit/test_cz_local_cleanup.py \
  tests/unit/test_native_block_optimization.py \
  tests/integration/test_native_block_physical_evidence.py -q
```

Negative regressions corrupt the merge result and exceed the raw proposal
budget. Neither may enter accepted output. Existing randomized, reversed-wire,
noncommuting, symbolic and protected-boundary cases exercise the reused rules.
