# Numerical automatic port partitioning

Simulation owns `classical_auto_partition.py`. It follows the unchanged Qualtran
v0.7.0 `AutoPartition.build_composite_bloq` chain in
`bookkeeping/auto_partition.py`, commit
`bcb17bb25360ed68ce3bc084645db3e929799389`, using native numerical callbacks in
place of builder operations. This is a composition of bound operations, not a
generic graph interpreter or a new dtype system.

Each `PackedValueBinding` contains an external name, bound numerical partition
ports and the existing packed-word codecs. The function partitions inputs in
declared binding order, separates child inputs from unused values, invokes the
child once and repacks outputs with the original unused values. The left-only
path returns child outputs without repacking. Constructor/layout legality,
including left-only restrictions and generated unused names, remains with
Compiler's existing `AutoPartition` metadata.

The implementation consumes shared `partition_classical_value` and
`unpartition_classical_values` directly. Replacement tests verify the split,
child and restore order and allow replacing the child without modifying its
consumers. The numerical composition does not expose foreign graph objects or
change the protected circuit API.

Reference tests compare every small input of four original Add wrappers: separate
packed words, interleaved unused storage, left-only outputs and a wholly unused
external port. The original unmodified composite's classical execution is the
reference. A measured two-bit Add wrapper additionally matches the original
primitive stream, all complex amplitudes for every packed input and spectator,
both unnormalized measurement branches, clean work and both QEC cost profiles.
The source metadata and numerical composition remain separate acceptance scopes.

Run `python -m examples.algorithms.classical_auto_partition_values` for the
ordinary `circuit.append(op)` path. It emits four T gates and one measurement
using one work bit, preserves an interleaved unused bit and a spectator, then
restores all inputs with the exact inverse. Seven wires are declared including
the spectator; the source model has four T equivalents, one measurement and six
peak logical wires. Partitioning contributes zero gates. Logical estimates and
emitted instructions are verified separately.
