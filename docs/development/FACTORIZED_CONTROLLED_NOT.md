# Factored controlled-NOT tensor leaves

The locked Qualtran v0.7.0 `CNOT.my_tensors` and the recovered `tensor.ipynb`
represent CNOT as two rank-three tensors: COPY and XOR, sharing one internal
edge. Native examples previously supplied a dense four-by-four CNOT matrix.
This batch ports the original factorization and supplies the missing pure-state
callback access to the existing graph allocator.

## Ownership and interfaces

Simulation owns `controlled_not.controlled_not_nodes`. Its ports are scalar
one-bit quantum ctrl and target, both THRU. COPY axes are `(ctrl-in, ctrl-out,
internal)`; XOR axes are `(target-in, target-out, internal)`, exactly as in the
source. The source arrays' complex zero/one data are preserved for the requested
Torch dtype/device. COPY reuses the existing coherent equality data in
`z_measurement_copy_node`; this use performs no measurement or tracing. The
source singleton tag names COPY/XOR use existing node names, without adding a
Core schema, tag registry or foreign object.

The caller supplies four distinct external labels and one graph allocator.
The leaf requests exactly one fresh internal edge and refuses reused, negative
or non-built-in-integer labels. Numerical labels are independent of physical
qubits and classical feedback slots.

`component_tensor_nodes_with_labels` exposes the existing allocator to pure
suppliers through a fourth callback argument. The original three-argument
`component_tensor_nodes` delegates through that entry without changing its
signature or callback contract. Both paths use the same `_component_tensor_nodes`
implementation, whose AST is unchanged. Full graph validation, SSA wiring, idle
identities, visitation checks, autograd and immediate pure-discard refusal keep
one owner. No second graph or contraction executor is introduced.

The executable `factorized_controlled_not.py` composes real existing CNOT
ProgramRecord bodies. Its explicit supplier verifies the declared primitive
body rather than guessing from names. The same supplier feeds pure and mixed
consumers. Compiler still owns graph construction and physical binding; numerical
factorization adds no physical instruction or workspace.

```bash
python examples/algorithms/factorized_controlled_not.py
```

## Exact source and consumer checks

Tests compare both nodes against unchanged source data, axis order and singleton
tag names over all 24 external-label permutations and four dtypes. Repeated
CNOT graphs (1, 2, 3 and 7 calls) compare complete pure and superoperator tensors,
complex spectator amplitudes and parameter gradients. Their physical consumer
uses permuted data wires, a spectator, an unused clean wire and the ordinary
`circuit.append(op)` path. Source and native physical behavior match, and unused
clean storage remains zero. CNOT has no measurement branches to enumerate.

The graph allocator is tested with repeated calls, idle words, zero-width and
empty-array idle signatures. Internal edges occur exactly twice and do not alias
future SSA or external edges. Supplier exception identity, old callback arity,
invalid traversal refusal and stopping at the first pure discard are retained.

| Repetitions | Source/native tensor nodes | Tensor elements / largest leaf | Actual primitive gates | Source QEC model / peak |
| --- | --- | --- | --- | --- |
| 1 | 2 | 16 / 8 | CX1 | Clifford1 / 2 |
| 2 | 4 | 32 / 8 | CX2 | Clifford2 / 2 |
| 3 | 6 | 48 / 8 | CX3 | Clifford3 / 2 |
| 7 | 14 | 112 / 8 | CX7 | Clifford7 / 2 |

The native tensor-node count follows the original two-node-per-CNOT recipe.
Additional physical work, measurement and feedback slots are all zero. Tensor
storage and contraction operations are not quantum gate counts; no speed or RAM
advantage over the prior dense leaf is inferred from this decomposition alone.

## Remaining scope

The recovered original `tensor.ipynb` matches frozen blob
`d91373e7d8eb21c48a0b870d99d1bb2973971861`. This batch enables its factorized CNOT
section, but does not claim the complete notebook, tensor-network drawing,
friendly index views, arbitrary tag sets or the full CNOT source module.
General flattening, provider discovery and wide-network simplification stay
open. The existing default-suite timeout is retained and not repeated for this
bounded change.

## Local verification

132 new cases pass. Concentrated regression gives 1,798 passes in 38.20 seconds
with 54 retained reference warnings; 84 CPU/tensor-network integration checks
pass in 4.94 seconds with two existing warnings. A final 132-case focused recheck
also measures the original graph's tensor-node totals. Strict typing covers all
five changed Python files; the final test-only source-node assertions pass a
further strict check. Black, Ruff, Simulation team scope, architecture,
multi-level IR and protected API checks pass. The three selected modules cover
142 statements and 50 branches at 97% combined coverage; the new CNOT leaf
covers all 20 statements and four branches. Two existing traversal profile
refusals remain uncovered in that measured selection, and the standalone
module-entry guard is executed separately.

Four locked source blobs match the frozen inventory. The existing native graph
allocator/traversal AST matches its baseline. Logs and resource evidence use
`g2-factorized-cnot-` in the reference workspace. This accepts the concrete
factorized CNOT capability, not a full source-file or notebook closure.
