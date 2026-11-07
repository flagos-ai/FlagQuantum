# Classical shaped partitions

Simulation owns `classical_partition.py`, ported from the unchanged Qualtran
v0.7.0 `_PartitionBase` classical methods in `bookkeeping/partition.py`, commit
`bcb17bb25360ed68ce3bc084645db3e929799389`. Compiler retains physical register
partitioning and layout legality. No foreign dtype or graph enters the numerical
boundary.

`PartitionValuePort` carries a bound name, word width, shape and existing scalar
and array codec callables. It does not replace register construction or add a
second shape-validation policy. Packed-word encoding is sliced in declared port
order using the source `int(np.prod(shape + (width,)))`. Scalar ports receive the
original bit slice. Shaped ports receive `np.asarray(slice).reshape(shape +
(width,))` before array decoding, preserving dtype inference and empty-axis
failure behavior.

Unpartition uses `np.asanyarray` and `ravel` on each supplied value, invokes the
port's array encoder, ravels each bit array and concatenates them. The existing
shared Join value boundary performs the final `tolist` and packed-word decode.
The original callback does not enforce supplied array shapes against metadata:
that diagnostic behavior is retained. Extra named values are ignored, missing
names raise KeyError, and codec/NumPy failures propagate without normalization.

Reference tests compare bit, signed/unsigned integer and fixed-point ports,
scalar/shaped/empty axes, input/output types, raw shape diagnostics and every
small packed word. Replacement tests prove that the shared Join and supplied
codecs are actually consumed without consumer changes. Compiler's existing
binding, complex-state and zero-cost resource tests remain in the regression.

Run `python -m examples.algorithms.classical_partition_values` for
`circuit.append(op)` with a 2x2 bit array and scalar flag. The example compares
complete complex amplitudes over all packed inputs and a spectator, then checks
inverse restoration. Partition/unpartition emits no gates, T, measurements or
work; the explicit cell operation emits one X. The source bookkeeping model has
zero gate cost and five logical wires; the example adds one spectator. Numerical
reshaping does not imply quantum allocation, release or stochastic execution.
