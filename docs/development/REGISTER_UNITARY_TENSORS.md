# Unitary register tensor leaves

Simulation owns this numerical bridge along the locked Qualtran v0.7.0
`GateWithRegisters.my_tensors -> _my_tensors_from_gate ->
tensor_data_from_unitary_and_signature` call chain. The reference commit is
`bcb17bb25360ed68ce3bc084645db3e929799389`; reference files remain unchanged.

`register_tensor.register_tensor_from_unitary` accepts a caller-supplied PyTorch
full matrix and a native signature. It reshapes every declared word as THRU,
selects output zero for LEFT ports and input zero for RIGHT ports, and retains
output-before-input, row-major word and MSB-first bit order. Data is not cast,
normalized or copied through NumPy. PyTorch retains its dtype, device and
parameter gradient graph. The existing `bind_register_tensor` supplies binary
labels; `ComponentTensorProviders` remains the explicit per-function supplier
boundary. No new provider registry, Core metadata or public export is added.

NumPy and PyTorch share one zero-slice constructor in `register_tensor_data`.
The existing raw NumPy entry still performs its original shape assertion,
THRU-signature construction, reshape, indexing and result-shape assertion in
order. Its dtype/subclass/scalar behavior and original failure boundaries remain
separate from the validated PyTorch entry. Native paired LEFT/RIGHT declarations
with a shared name retain the existing validated-name support. This does not
claim that the source raw signature constructor accepts those declarations.

The caller declares matrix semantics. This projection does not numerically test
unitarity or infer a missing phase. Use the existing Simulation `get_unitary`
owner for native executable unitary circuits; its existing non-unitary and size
refusals remain. Dense matrix storage is exponential in all declared bits. This
is a deliberate dense leaf, not a fallback for a large factorized Add graph or
measured AND erasure. Projection onto a clean absent port is neither trace nor
reset. Zero-width words retain size-one word axes; empty arrays retain no axes.

Run the example:

```bash
python examples/algorithms/unitary_register_tensors.py
```

It replaces a one-bit Add CNOT supplier with the existing native unitary owner,
register projection and binder, without changing the typed graph. General
primitive suppliers use the same numerical boundary, including phase-bearing
matrices; Compiler still owns graph selection, order and physical binding.
Source-wide provider discovery and complete notebook closure remain open.

## Verification and resource accounting

The 437 new cases cover all 27 three-port direction combinations, scalar,
row-major array and empty-array ports, complex64/complex128 data and binary label
order against the unchanged `_my_tensors_from_gate`. Gradients match finite
differences through the original NumPy projection; an actual native trainable
H/Rz/H circuit also retains its parameter derivative. Zero-width word axes,
paired native port names and invalid classical/width/shape/matrix inputs are
checked explicitly. All 256 angles `2*pi*k/256` run through a real H/Rz/CNOT
component provider and ordinary `circuit.append(op)`, preserving full complex
amplitudes with a spectator and permuted physical binding. No phase quotient or
classical truth-table replacement is used.

Concentrated regression and its coverage run each pass 1,342 cases with 15
existing reference numerical warnings. CPU/TN integration passes 84 cases.
The two numerical owners cover 106 statements and 32 branches at 100%. Strict
mypy, Black and Ruff pass all four changed Python files; Simulation scope,
architecture, internal IR, protected API and the executable example pass. Five
pinned source blobs match the immutable inventory. The original raw projection
tests and existing measured Add branch/work/feedback/resource tests also pass.

| Quantity | Observed or estimated result |
| --- | --- |
| Actual emitted component gates, every angle | H: 1, Rz: 1, CX: 1 |
| Additional work / measurement / feedback slots | 0 / 0 / 0 |
| New gates from numerical projection | 0 |
| Original QEC model, 4 angles | 3 Clifford, default estimated T count 0 |
| Original QEC model, 1 angle | 2 Clifford + 1 T, default estimated T count 1 |
| Original QEC model, 251 angles | 2 Clifford + 1 rotation, default estimated T count 11 |
| Original QubitCount model, every component | 2 logical qubits; the spectator is external |

These are the locked model's unmodified classification buckets, not a statement
that the physical Rz was synthesized into T gates. Its default rotation synthesis
estimate is separate from actual primitive counts. The numerical bridge adds no
physical gate or resource estimator. Native invalid-matrix/classical-port
ValueErrors and `get_unitary`'s non-unitary refusal are explicit native boundaries,
not claimed to match Cirq's unitary-discovery exception classes. The preceding
default-suite 600-second timeout remains open and was not rerun. No source parent
is closed by this extension; source-wide discovery and full notebook acceptance
remain open. Evidence prefix: `g2-unitary-register-`.
