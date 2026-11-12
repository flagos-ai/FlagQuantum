# Integer basis vector tensor suppliers

Simulation's `tensor_network/register_basis_vectors.py` supplies one rank-one
ket or covector per existing bit label. It follows pinned Qualtran v0.7.0
`_IntVector.build_composite_bloq` -> integer/bit codec -> `_ZVector.my_tensors`.
The source is unchanged at `bcb17bb25360ed68ce3bc084645db3e929799389`.

`register_basis_vector_nodes(signature, connections, values, like, state=True)`
accepts a pure-RIGHT quantum signature for kets, or pure-LEFT with `state=False`
for effects. Connections contain row-major words of MSB-first integer labels.
Values are unsigned basis indices, independently of the declared numeric dtype,
as in source IntState/IntEffect. Shaped values use `asarray` and indexed traversal;
there is no shape broadcasting or conversion through the port's field codec.

The existing numerical vector validator retains constructor-before-codec order.
Width-one words use the shared bit encoder; other words use the shared unsigned
encoder. The existing label binder validates finite dimensions and connections.
Every label must be distinct. Missing values and array-index errors propagate.
Native validation messages remain native; no foreign exception types are added.

`register_zero_state_nodes` now selects all-zero words with `values=None` through
this owner, without allocating an array of zero values. Existing
component suppliers, graph traversal, contraction and gradients remain consumers
of the same `TensorNetworkNode` records. No Compiler dependency, registry, graph
interpreter, Core schema or public export is introduced. Compiler continues to
own physical integer preparation and effect bindings. Users still append the
existing operation with `circuit.append(op)`.

Unlike the source's zero-width IntVector join/split failures, the established
native zero-state boundary emits no nodes for zero-width words or empty arrays.
The new supplier retains that explicit boundary. It does not create an implicit
scalar identity for an empty contraction. Classical ports and mixed directions
are refused before inspecting values. Symbolic width/shape and automatic
`flank_with_classical_vectors` graph construction remain outside this supplier.

## Evidence and resource interpretation

Tests compare every 1–4-bit integer in both directions with the original
IntVector decomposition, leaf tensor order and contracted vector. Array scalar,
vector, matrix and empty shapes retain row-major bit order. An original H then S
circuit flanked by IntState/IntEffect agrees for all four input/output choices,
including complex and negative phase; replacing the operator with a trainable
phase preserves the exact autograd derivative.

The existing zero-state component consumer is exercised through a replacement
supplier. Its complex states, gradient and measured addition forward/inverse
histories retain their previous evidence, including clean work bits. The new
physical consumer appends the existing integer preparation on all 1–4-bit values,
checks full complex output, zero spectator and double-append restoration.

Tensor supply emits zero quantum primitives and allocates no work qubits.
Physical integer preparation emits `popcount(value)` X gates (0–4 in the tested
range), zero T and zero measurements, with no work allocation. Its independent
upstream QEC model estimates **one arbitrary Clifford**, rather than counting
those X gates. These are separate accounts. Integer preparation is coherent,
with one deterministic branch; measured Add histories belong to the unchanged
consumer regression, not this vector supplier.

This supplies a prerequisite of classical/tensor cross-checking. It does not
claim the full `xcheck_classical_quimb.py` graph transformation has been ported,
close the source z_basis module or start G3.

Local verification: 54 new tests, 4068 concentrated tests and 84 CPU/TN tests passed. Strict typing of three files, Black, Ruff, Simulation scope, architecture, internal IR and API protection passed. The basis supplier and replaced zero supplier each have 100% statement/branch coverage. Eleven locked source blobs matched the inventory.

Bounded default smoke/unit: `7 failed, 22385 passed, 300 skipped, 1048 warnings in 186.78s (0:03:06)`. All observed failures belong to the seven-case baseline; this is not a green full suite. Local evidence prefix: `g2-register-basis-vectors-`.
