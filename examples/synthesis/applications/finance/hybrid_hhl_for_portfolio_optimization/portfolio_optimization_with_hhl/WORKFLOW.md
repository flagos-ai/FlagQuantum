# Reversible matrix HHL

The native Python qfunc prepares the source vector, allocates a fractional
phase counter and indicator, then expresses QPE in `within()` and a raw-word
rotation table in `apply()`. Compiler generates the inverse QPE. The indicator
is allocated before the pure region, which preserves its fresh-zero state
while keeping allocation out of reversible operations. Users specify matrix,
rotation angles and quantum types; no physical wire numbers or gate design
are needed.

`physics.py` retains every source matrix entry and rotation angle. The tutorial
with an unnormalized rounded vector uses explicit approved normalization;
its unchanged source remains refused. The hybrid portfolio example reuses a
length-polymorphic overlap helper and drops its phase/data/comparison names.
Drop retains the full environment without reset, uncompute or reuse. Named
probabilities marginalize that environment. Full-vector validation includes
all dirty environmental bits separately from matrix decomposition scratch.

The independent Circuit comparison uses Householder preparation, controlled
matrix powers, dense Fourier, direct rotation-table blocks and direct Fredkin
gates where applicable. It compares source/native/full reference vectors
without global-phase alignment, two nonzero parameter variations, source bit
order, all scratch at zero and actual width/operation/depth/memory refusals.
The four-bit-data portfolio requires three clean work bits for matrix
decomposition; the other three use none. Resource equality or improvement is
not assumed. Complete Notebook workflows remain pending.

Current native QPE admits one static matrix call in a zero-argument lambda,
an allocated unused one-to-eight-bit all-fractional counter and root pure
within/adjoint composition. Rotation tables admit one-to-eight distinct raw
selector bits, X/Y/Z axes, 2**N finite real angles and a disjoint writable target.
General controlled tables, arbitrary QPE callbacks and joint qstruct-field
state preparation remain separate dependencies.

Measured native costs: 10 physical bits, 0 clean work bits, 449 operations and depth 412. The original uses 429 operations and depth 402; the native controlled-swap decomposition adds 20 operations and 10 depth. This remains an explicit optimization gap.
