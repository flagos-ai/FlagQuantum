# Signed block SELECT

G2-BE06 ports the signed child and padded SELECT dependency of the unchanged
Qualtran v0.7.0 `LinearCombination`, commit
`bcb17bb25360ed68ce3bc084645db3e929799389`. Parent source blob:
`a3d8aa7fafa7a6f3fcaf095826751209427cd255`. Attribution is retained.

`LinearCombinationSelect` belongs to algorithms/primitives. It composes shared
`AutoPartition`, `indexed_component_ir`, lazy bound-operation capture and
`indexed_component_cost`. Child system widths must agree and signal states
must be identity preparation. Ancilla and resource widths take their respective
maxima. Each child receives its original leading ports; remaining tails are
unused bystanders, including arbitrary nonzero and entangled states.

Address traversal uses the number of original terms, not the padded PREPARE
row count. Source bounded-tree padded-address aliases are retained, rather than
silently redefining them to identity. A positive external control is optional.
Child factories supply their directional positive-controlled native operations,
preserve the predicate and clean private work. Repeated child calls may reuse
a private pool; parent traversal work remains disjoint while predicates live.
No coherent control is mechanically added to measurement/reset instructions.

Negative coefficients insert the original controlled global phase, Z on the
predicate. The actual controlled source graph visits the controlled child then
Z-power, in both directions (exponent +1/-1). This differs from reading the
uncontrolled Phase constructor text in isolation. Gate export and native
composition both retain the actual controlled order. No four-instruction scalar
carrier is needed for this controlled sign.

Resource queries require a controlled child model per term. Each model retains
its own work and the parent accounts for padded bystanders and live predicates.
Both original QEC profiles are supported; no child model is inferred from IR.
The caller must choose models corresponding to the supplied child path.

Validation covers 24 original gate configurations: two/three/four terms, mixed
ancilla/resource widths, both directions and external control present/absent.
Primitive operations, operands, order on shared wires and measurement feedback
match. Both QEC profiles and original peak width match (48 model comparisons).
Eight original full matrices are tested on arbitrary complex states across all
ports, including control superpositions and padded addresses, with every
measurement history and private work retained in the amplitude comparison.

Eight further gate/matrix configurations use controlled one-bit Add children
and reuse one private pool across terms. All small measurement histories match
absolute complex source amplitudes, and scratch clears. Forward declared costs
match, including the existing source negative-Clifford one-bit Add anomaly.
The generic controlled adjoint follows its original one-CNOT-to-Toffoli path,
using the existing controlled-not primitive; substituting specialized CAdd-adjoint
would change the source gate stream. Export keeps Toffoli atomic on both sides
for that path. Four original adjoint cost queries fail at atomic controlled
And-adjoint, although the complete circuit and source matrix are executable.
Those failures are asserted explicitly, not replaced by Toffoli estimates.

Run:

```sh
python -m examples.algorithms.linear_combination_select
```

The three-term X,-Z,X example uses `circuit.append(op)`, coherent populated and
padded addresses and a measured predicate. It verifies the full complex state,
clean work and exact inverse. Model and emitted counts are printed separately:
4 T equivalents / 4 T or T-dagger, one measurement and four logical qubits.

Additional contract tests reject wrong widths, overlapping work, missing child
groups, escaped child wires, invalid control counts/directions and invalid term
lists. Strict typing, formatting and architecture checks accompany the focused
regression. The full parent PREPARE–SELECT–PREPARE-adjoint composition remains
open; this slice does not certify every possible child control specialization.
