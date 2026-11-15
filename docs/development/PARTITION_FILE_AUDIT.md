# Partition source files

Reference: Qualtran v0.7.0 at
`bcb17bb25360ed68ce3bc084645db3e929799389`. The frozen bookkeeping
files are Apache-2.0, copyright Google LLC.

| Item | Source file | Frozen Git blob | Disposition |
| --- | --- | --- | --- |
| Q-0327 | `partition.ipynb` | `65ff47688d0c7715f2df9f95a97ae3d34b21dd8b` | Closed by the complete Partition and AutoPartition tutorial. |
| Q-0328 | `partition.py` | `8cebbb58095d172f3db564cdfc231537e6214957` | Open for the full external dtype and wire-symbol protocols and direct object interop. |
| Q-0329 | `partition_test.py` | `c6c3236332577115ff6f09e17f340fe79825e1f4` | Closed for the original runnable scenarios and both source tutorials. |
| Q-0332 | `split2.ipynb` | `eaf039d7f204f9a1e4fa1cb6d26fb2d93c73aae8` | Closed by the complete Split2 and Join2 tutorial. |

The Compiler shares one typed Partition builder between direct callers
and AutoPartition. Its declaration places the packed word `x` first
in either direction. The named parts occupy contiguous slices in
declaration order, each array follows row-major element order, and each
word is big-endian. The inverse swaps port sides and preserves their
order. Both directions reuse the typed split/join builders. Binding
rewires existing words without gates, measurements, or work allocation.
Symbolic declaration and concrete physical binding use separate paths.

The shaped original tutorial binds six two-bit elements and one 37-bit
word on 49 lines. Both directional signatures and all bound wires match
the locked example, including a noncontiguous-in-order caller binding.
The source model has zero gate cost and a logical peak of 49. The native
binding emits zero gates. The tutorial retains both original AutoPartition
scenarios and the direct graph, then appends a bound controlled exchange.
The Split2/Join2 tutorial retains both symbolic signatures and adds exact
wire and integer round trips.

The original 12-line wrapper around the test-only multi-register child
preserves the Cirq wire map and the source tensor factorization. The
native test binds the original one-bit port, four two-bit array elements,
and three spectator bits. It appends X and four CNOTs corresponding to
the test atoms' defined tensors and compares all 4,096 complex amplitudes
with a contraction of the unchanged source tensor network. The test
atoms provide tensor callbacks. Their arbitrary test cost metadata is
distinct from the physical X/CNOT stream.

The earlier tests cover scalar and shaped codecs, signed/fixed-point and
finite-field partition values, small integer round trips, zero-width
wire slices, and the source's invalid empty or mismatched layouts.
The source's raw Cirq Join concatenates direct keyword arguments in
caller insertion order. Its typed composite and tensor path uses the
declared order. The native named path follows the declaration, and a
regression test preserves the direct raw exporter discrepancy.

The unchanged non-notebook source suite has twelve passes and nine
fixture-mode skips. The original 17- and 14-cell notebooks and the
native 19- and 16-cell versions passed independent kernels. The native
runner rejects reference imports and validates the five SVGs.
