# Tensor-product block encoding

G2-BE03 owns the locked TensorProduct wrapper in algorithms/primitives. Child
metadata uses the existing private structural view; this view moves to a shared
module because both Phase and TensorProduct consume it. Compiler RegisterLayout
and partition_register bind packed big-endian system/ancilla/resource words.
Child factories receive only their partition, private work, result slots and
direction, and return an existing bound Circuit operation. No child arithmetic,
SELECT or phase decomposition is duplicated.

Capture uses the existing lazy Circuit builder to read child IR without executing
it. Validate each child's gate operands against its own partition and work before
constructing the parent operation. Work may be reused between sequential children
but must not overlap any through port. The child owns cleanup. Forward calls use
source tuple order; adjoint calls reverse it and request directional children.
Costs are explicit child models: each call carries every other child's through
ports as bystanders; transient child work contributes a maximum, not a sum.

Metadata follows the original formulas: system/ancilla/resource widths add,
alpha is the product of child alphas, and epsilon is sum(alpha_i * epsilon_i).
Nested grouping is preserved rather than flattened to a different error formula.
Symbolic alpha/epsilon expressions are retained. Register binding is concrete in
this slice. Signal preparation is identity only when every child uses the
existing canonical identity preparation; the same nonidentity case raises
NotImplementedError in the untouched source and native wrapper.

There are 64 complete original gate-stream configurations spanning single,
repeated, heterogeneous and three-child blocks; T, CNOT, two-/three-bit Add;
mixed auxiliary layouts; both directions; and shared or dedicated child work.
Both QEC profiles match all seven categories and source logical peak for every
configuration (128 model comparisons). Two further original flows verify
noncontiguous packed ports. Twelve full source tensor cases apply every small
measurement history to arbitrary complex amplitudes over all through ports,
including cross-block entanglement, and clear all child work. Two nested tensor
cases additionally compare original gate dependencies, full complex matrices,
measurement branches and source resource models while retaining inner work.

Source peak accounting is distinct from physical storage reserved by a caller:
two two-bit Add blocks have 8 through ports and a source logical peak of 9.
A caller dedicating one work wire to each block reserves 10 wires; sharing the
clean work reserves 9. No execution-memory peak is inferred from either number.
The independent child cost plus other through ports is charged for each call;
summing private child work would misstate the source estimate.

The existing Circuit constructor/append/to_ir boundary is lazy. A refusal probe
on Circuit.state and Circuit.run proves this composition does not enter either
numerical execution path. Child factories see source partitions in tuple order,
or reverse order for the adjoint. Invalid packed widths, overlap with through
ports, duplicate result slots and child access to another partition are refused.
The caller supplies and owns directional children and cleanup; capture does not
try to invert measured primitive streams. Model queries do not invoke factories.

Run `python -m examples.algorithms.tensor_product_block_encoding`. The Add/T
example uses `circuit.append(op)`, verifies the phase-sensitive full state and
an exact inverse, and reports separately:

| Quantity | Source model | Emitted IR |
|---|---:|---:|
| T equivalents / T instructions | 5 | 5 |
| Measurements | 1 | 1 |
| Logical peak / reserved qubits | 6 | 6 |

There are 93 new tests and 235 concentrated passes without skips, including the
existing phase, unitary, identity preparation, SELECT and LCU consumers. Strict
mypy covers the shared metadata view, both consumers and the new example;
Black/Ruff, ownership and architecture checks pass. Source blob
`d1a01a8eccee2fff1f319668b6bd34f4a3890d13` matches the pinned tree and installed
reference. No source or protected API is modified. Symbolic register binding,
nonidentity signal tensor preparation and whole-file acceptance remain open G2
items; zero-wire scalar representation retains the prior explicit boundary.
