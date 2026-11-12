# Source-ordered conjunction tensors and ladder composition

Simulation owns `conjunction_tensor_node`, a direct port of Qualtran v0.7.0
`And.my_tensors` at commit `bcb17bb25360ed68ce3bc084645db3e929799389`.
The original five axes are preserved: two incoming controls, two outgoing
controls, then target. The data has four unit entries in a `(2,2,2,2,2)`
complex tensor, with dtype/device following the caller's `like` tensor. Adjoint
changes only target direction and the original node name. It is a rectangular
logical effect, not a reset, trace or physical measurement channel.

The executable `examples/algorithms/conjunction_tensors.py` composes the original
MultiAnd ladder using the already authoritative `multi_and_stages` and real
`control_predicate_program` bodies. Each exact definition has an explicit
activation binding; altered or missing definitions fail rather than selecting a
fallback. No numerical dispatch depends on foreign objects or parsing names.

```python
import torch
from examples.algorithms.conjunction_tensors import (
    conjunction_program, conjunction_providers,
)
from flagquantum.compiler.registers.component_graph import iter_component_calls
from flagquantum.simulation.tensor_network.component_dense import contract_component_tensors

values = (1, 1, 1, 1)
record = conjunction_program(values)
like = torch.empty((), dtype=torch.complex128)
providers = conjunction_providers(values, like)
order = tuple(i for i, _, _ in iter_component_calls(record, "multi_and"))
matrix = contract_component_tensors(record, order, providers.supply, like, "multi_and")
assert matrix.shape == (128, 16)
```

Ports use a row-major `ctrl[n]` bit array, followed by `junk[n-2]` and scalar
target outputs. For two controls, no empty junk port is added. All `n-1` prefix
outputs remain live logical results owned by the caller. They are not zero work
at forward exit. Physical binding uses an explicit clean pool and
`allow_workspace_outputs=True`, transferring those outputs to the caller.
Existing measured reverse-ladder erasure consumes the correct computed prefix
values and clears their storage. Circuit execution still uses
`circuit.append(op)` with existing operation objects and native instructions.

Tests compare all 28 activation patterns for two through four controls against
unmodified source. They compare every leaf's data/name/axis incidence, complete
rectangular matrices, exact physical instruction dependencies and operands,
complex superpositions entangled with an untouched spectator, gradients and all
small erasure histories (up to eight branches). Every postselected branch retains
its unnormalized amplitude `input/sqrt(2**(n-1))`; every prefix bit is zero after
erasure. The original rank-five provider and an existing classical-action table
provider are exchanged without changing the graph consumer or physical program.
Four dtypes and both directions cover the standalone leaf.

## Resource accounting and limits

The original notebook uses four positive controls. Its native/source tensor
network has three rank-five And leaves. The dense matrix has shape `(128,16)`;
node shape is not a physical work count. Actual primitives are recorded separately
from source QEC model estimates in the batch evidence. Three prefix qubits are
logical outputs; there is no additional scratch. Measurement-assisted erasure
uses three measurement slots and restores those outputs, and the source model's
measurement estimate remains a separate field.

This accepts the And/MultiAnd composition dependency of the original tensor
notebook, not its whole 18-cell workflow. Bell/custom-provider notebook cells,
contraction-info presentation, general flattening and Quimb layout equivalence
remain open. Symbolic control lengths and the source's two-control empty-junk
port are outside this concrete example. No protected Core API, compiler emitter,
resource model, required dependency or execution fallback changes here.

| Four positive controls | Actual primitives | Locked source QEC model |
| --- | --- | --- |
| Forward | H6, T6, Tdg6, CX18, S3 | And3 (12 T equivalents), peak7 logical qubits |
| Erase | H3, measure3, CZ3, reset3 | Clifford3, measurement3 |

For `(1,0,1,0)`, both physical directions add four X gates; source forward
adds Clifford4 and source erase reports Clifford7 plus measurement3. Native
reset explicitly clears measured storage; the QEC model is not a literal reset
instruction count. No gate-count equivalence is inferred across these levels.

## Local validation

67 new focused cases, 1,940 concentrated regression cases and 84 CPU/TN checks
pass. Strict typing covers all three changed Python files. Black, Ruff,
Simulation-team scope, architecture, multi-level IR and protected API checks
pass. The supplier has 100% coverage; supplier/example combined coverage is 98%
(78 statements and 14 branches). Reference contraction emits numeric warnings
while the exact-array assertions pass; those warnings remain visible. The known
default-suite timeout was not repeated and whole-repository success is not
claimed. Source blobs are `2e6efeb53938ac43f6f0330d79140f04138861eb` (And)
and `d91373e7d8eb21c48a0b870d99d1bb2973971861` (notebook), unchanged.
