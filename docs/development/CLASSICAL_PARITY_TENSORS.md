# Classical parity tensors (GRAPH75)

Simulation owns `tensor_network/classical_parity.py`. It supplies the selected
`Parity(n)` value mapping on existing numerical edges, consuming the Core
signature `x:CBit[n] THRU, parity:CBit RIGHT`. The unchanged value reference is
Qualtran v0.7.0 `gf2_multiplication.py::Parity.on_classical_vals`, pinned at
bcb17bb25360ed68ce3bc084645db3e929799389: return x and `np.sum(x, dtype=int) & 1`.
Copyright/source attribution is retained in the implementation.

## Ownership and layout

`classical_parity_nodes(count, incoming, outgoing, fresh_label, like)` consumes
row-major, MSB-first binding maps and returns existing `TensorNetworkNode`s.
The owning mixed graph supplies `fresh_label()` from its shared SSA namespace.
The helper validates external bindings before allocating internal labels, then
rejects invalid, repeated or colliding allocated labels before constructing data.
Empty input returns the parity-zero ket without looking up x bindings or calling
the allocator. No foreign graph, registry, physical classical slot, decomposition
or automatic function-name discovery is introduced.

Each of count chain nodes has axes `(x-out, carry-out, x-in, carry-in)` and entries
one exactly when x is preserved and carry-out equals x XOR carry-in. A zero ket
fixes the initial carry. The final carry is the parity result. This is an exact
native factorization of the source value mapping, **not an upstream tensor or
gate recipe**. Locked Parity has neither a tensor implementation nor an executable
quantum gate decomposition. `Parity.my_tensors` raises `NotImplementedError`;
this negative reference is retained in every value-table scenario. Original
classical-array table enumeration also uses zero quantum width and cannot serve
as a CBit tensor supplier. Those reference failures are not patched or hidden.

Dtype/device follow like. All count step nodes share one 16-element tensor;
the zero ket has two elements. Logical node elements total `16*count+2`, while
unique tensor storage is 18 elements for positive count and two for zero.
Internal integer labels number count, nodes count+1. These are returned-storage
sizes, not peak Python construction or contraction memory. Explicit dense
inspection has `2**(2*count+1)` elements and must remain a small-system test.
No physical gate, XOR instruction, workspace, reset or sampling is emitted.

The coherent truth table alone does not dephase CBit axes or produce independent
classical records. The mixed channel owner must provide measurement environments
and explicitly discard selected legs. This helper does not infer those effects.

## Evidence and supported workflow

Four dtypes and widths 0/1/2/3/5/8 compare the entire contracted table with
unchanged `Parity.call_classically` for every input, including its forwarded x.
Tests verify shared storage, zero-input parity and allocator/binding refusals.
Combination with source-compatible classical controls is described in
[controlled tensor nodes](CONTROLLED_REGISTER_TENSORS.md#classical-feedback-nodes-graph76).
There all small measurement branches retain complete complex amplitudes,
relative-phase derivatives, spectator entanglement and clean workspace.

Users still append the bound existing operation using `circuit.append(op)`.
The graph wrapper and physical Parity implementation already exist in Compiler;
this numerical leaf does not change them. A runnable existing user workflow is:

```bash
python -m examples.algorithms.binary_field_arithmetic
python -m pytest tests/unit/test_classical_parity_tensors.py -q
```

Its measured GF(8) product erasure retains the existing physical parity/XOR and
controlled-CZ stream. Actual instructions and upstream resource-model estimates
remain distinct. This is a `single_device_fast_path` inspection path; no GPU,
distributed capacity, full source-file parity or new public API is claimed.
