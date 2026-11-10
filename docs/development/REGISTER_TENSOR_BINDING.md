# Register tensor binding (GRAPH57)

`simulation/tensor_network/register_tensor_binding.py` owns finite register-axis
binding and pure-state output grouping. It translates the unchanged Qualtran
v0.7.0 `_dense.py::_order_incoming_outgoing_indices` and the pure-state branch of
`_group_outer_inds`/`quimb_to_dense` onto existing native integer tensor labels.
Source is locked at bcb17bb25360ed68ce3bc084645db3e929799389. Copyright and
source references are preserved. Foreign connection objects and Quimb remain in
the reference tests; production imports neither.

The existing `TensorNetworkNode` and `_contract_nodes_greedy` already own node
data and contraction. Their integer labels avoid the one-character limit of
the older `ContractionGraph` convenience interface. This batch does not add a
graph representation, label allocator, contraction engine, compiler operation,
backend selector or public API.

## Interface, bit order and ownership

`register_tensor_label_groups(signature, incoming, outgoing)` accepts existing
Core register metadata. Each side maps a port name to a tuple of words in
row-major order. Each word is a tuple of the caller's existing bit labels in
MSB-first order. Even a scalar word uses one outer tuple. For example, a scalar
two-bit input is `{"q": ((100, 101),)}`; a two-element one-bit array is
`{"q": ((100,), (101,))}`.

The resulting groups contain all output bits followed by all input bits, in
signature order. Empty groups are omitted. Zero-width and empty-array ports
require no binding, matching the source's lack of a connection lookup when no
bit is visited. Additional mapping keys are ignored. Missing live ports raise
KeyError; mismatched word/bit counts and noninteger/negative labels raise
ValueError. The caller owns graph connectivity; these labels do not imply
physical-wire ownership or clean/borrowed allocation. Internal shared labels
remain shared, with no renaming or deduplication.

`bind_register_tensor(data, signature, incoming, outgoing)` follows the source
protocol's binary reshape and returns the existing TensorNetworkNode. Dense
data must already follow the register output/input axis order. Its element
count must match the declared bit axes. PyTorch dtype, device and autograd
connectivity remain intact; reshape can copy noncontiguous input according to
normal PyTorch semantics. No conversion through NumPy occurs.

`contract_register_tensors(nodes, signature, incoming, outgoing)` supplies the
ordered output labels to the existing greedy contraction owner and groups its
result into a scalar, vector or matrix. A scalar-only network supplies an
explicit scalar node. An empty node list is rejected rather than creating a
hidden tensor with an inferred dtype/device. Superoperator grouping, automatic
component flattening and Quimb's simplification heuristic remain outside this
finite pure-state slice. Dense outputs retain their exponential storage cost.

## Existing workflow and verification

User operations still enter through `circuit.append(op)`. Four activation
patterns compose a controlled X followed by a controlled Z. Native Pauli child
suppliers produce the controlled tensors from GRAPH56, then the new binder
connects their intermediate axes. The resulting operator matches actual
unchanged source `my_tensors` and `quimb_to_dense`, with source external-index
ordering. Replacing the first native tensor with identity changes no consumer.

The same two logical operations use the existing measured predicate emitter.
Their primitive names, wires, parameters and feedback are checked against the
original source decompositions. All sixteen forward measurement branches
(four patterns times four histories) preserve complete complex joint states
and the entangled spectator, with clean work zero. All sixty-four forward/inverse
history pairs restore the input with the correct unnormalized branch weight.
Sixteen forward and sixteen forward/inverse append trajectories also agree.

Each direction emits eight T/Tdag and two measurements/resets, reusing one
clean work wire and one result slot. Independent source models count eight
T-equivalents and two measurements; full primitive counts and all seven source
QEC categories are recorded separately. Five caller wires include the spectator,
while source logical peak is counted separately. Binding/contraction itself
emits no gates and is not a logical-resource estimator.

Twenty-seven layout cases compare the exact source index sequence and dense
output grouping across scalar, matrix-shaped and empty arrays, LEFT/RIGHT/THRU
ports, and a zero-width word. Source outer indices are deliberately presented
in reverse order. A real PyTorch rotation contraction preserves the original
storage on its contiguous reshape and gives the analytical gradient. Invalid
bindings fail before the contraction engine is called.

Run from the repository root with the locked reference environment:

```bash
python -m pytest tests/unit/test_register_tensor_binding.py -q
```

## Batch evidence

Implementation: abe56fa3. The 39 new scenarios pass. Related regression passes
697 tests in 6.02 seconds with two existing warnings. Focused coverage passes
107 tests in 3.64 seconds: all three register tensor owners have 100% statement
and branch coverage (114 statements, 48 branches). All 64 integration tests
pass in 4.18 seconds. Strict mypy covers the three owners and three test files;
Black, Ruff, Simulation scope, architecture, Program and protected API checks
pass. Five unchanged upstream `_dense_test.py` tests pass in 1.81 seconds.
Six source Git blobs match the original inventory.

Default smoke/unit stops at maxfail with 21400 passed, 300 skipped, 1037 warnings
and five failures in 109.68 seconds. Those five IDs belong to the independent
seven-failure baseline; the other two were not reached before interruption.
This is not a whole-suite success claim. Logs and eight forward/inverse resource
scenarios are retained at ftqc-reference-env/g2-tensor-binding-*.

G2 remains open. Next is the original tensor-flattening precedence between a
custom tensor and a decomposition, including failure handling, using existing
component traversal. Open-system grouping and complete parent-file acceptance
remain separate obligations. No CI wait or automatic PR merge is implied.
