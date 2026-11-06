# Packed preparation ports

`compiler/registers/partition.py` binds named slices of a caller-owned register.
The tuple order defines the layout, and each slice retains its input bit order.
Flatten shaped ports in row-major order before binding. Empty slices are allowed.
The binding emits no gates and allocates no work qubits.

`algorithms/primitives/black_box_prepare.py` binds the child preparation to a
packed selection word and a packed logical junk word. The caller supplies the
workspace and classical slots separately. Forward and adjoint calls use the same
physical binding. A zero-width selection returns an empty instruction stream,
following the locked source shortcut.

For alias sampling, the junk order is sigma, sparse index (sparse only), alt,
keep, and less-than-or-equal. The reflection consumer uses the child's adjoint
and forward emitters on those same registers. Logical junk may remain live
after forward preparation and is cleared by its adjoint on prepared states.

The tests read ordered ports and complex amplitudes from the unmodified
Qualtran v0.7.0 source. They compare the packed and direct gate streams in both
directions, verify work cleanup in measurement branches, and run a reflection
with the packed alias preparation. The wrapper adds zero primitive gates,
measurements, or work qubits. Resource requirements come from the child.

This binding covers concrete through ports. General AutoPartition operations
with left-only, right-only, or unused ports and symbolic widths remain open.

Sources:

- [Partition](https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/bookkeeping/partition.py)
- [AutoPartition](https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/bookkeeping/auto_partition.py)
- [BlackBoxPrepare](https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/state_preparation/black_box_prepare.py)
