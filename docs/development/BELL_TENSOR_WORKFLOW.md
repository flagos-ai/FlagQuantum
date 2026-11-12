# Bell states, custom tensors and explicit zero initialization

`examples/algorithms/bell_tensor_workflow.py` composes existing native owners to
execute the corresponding Qualtran v0.7.0 `simulation/tensor.ipynb` workflows.
There is no new graph, numerical backend, compiler emitter, Core schema or
provider registry. This is a local single-device diagnostic example.

```python
import torch
from examples.algorithms.bell_tensor_workflow import bell_program, contract_bell_program
from flagquantum.compiler.registers.component_zero_initialization import initialize_component_from_zero

like = torch.empty((), dtype=torch.complex128)
prepared = contract_bell_program(bell_program(prepared=True), like)
circuit = bell_program(prepared=False)
matrix = contract_bell_program(circuit, like)
initialized = initialize_component_from_zero(circuit, "bell")
state = contract_bell_program(initialized, like, entry="from_zero")
torch.testing.assert_close(prepared, state, rtol=0, atol=0)
assert matrix.shape == (4, 4)
```

The first path follows the notebook's `BellState`: PlusState, ZeroState, CNOT.
The second follows H/CNOT as a two-input unitary and then the original
`initialize_from_zero` wrapper. Ports are ordered q0/q1 for direct preparation
and q1/q2 for the unitary, with MSB-first bit axes. Compiler owns allocation,
root wrapping and full/partial expansion; Simulation owns explicit data suppliers,
fresh numerical labels, unitary projection and contraction. Numerical labels
are not physical qubits. Output qubits remain live owned results, not scratch.

The example reuses the canonical integer-ket allocation/split/join scaffold for
PlusState, replacing its single X with the actual H preparation. This preserves
its primitive body and source leaf selection without introducing another state
builder. Zero allocation uses the existing exact canonical definition check.
Providers refuse altered known definitions. No whole-Bell tensor is registered;
bypassing expansion fails with the existing missing-provider error, as the
unmodified source fails when `cbloq_to_quimb` sees an unflattened Bell root.

## Supplier replacement and source evidence

With `factorized_cnot=False`, the same graph consumes a dense CNOT tensor in the
notebook's `MyCNOT` output-before-input order, obtained from the existing native
unitary owner. The default uses the exact COPY/XOR leaves and graph-owned fresh
labels. No consumer or physical definition changes between those choices.
Tests execute the original notebook definition cells unchanged after checking
blob `d91373e7d8eb21c48a0b870d99d1bb2973971861`, from locked commit
`bcb17bb25360ed68ce3bc084645db3e929799389`. They compare the original MyCNOT
matrix, original BellState and H/CNOT circuit, both explicit zero-state paths,
full/partial expansion and complex64/complex128 results.

Direct prepared networks contain three dense nodes or four factorized nodes;
CNOT contributes one dense node or two COPY/XOR nodes. These are representation
counts, not different physical programs. Bell's state is
`[1/sqrt(2), 0, 0, 1/sqrt(2)]`. The H/CNOT circuit remains a 4x4 unitary until
inputs are explicitly initialized. Complete complex superpositions entangled
with a spectator, gradient propagation and physical inverse restoration are
verified, including absolute phase.

| Workflow | Actual emitted primitives | Original QEC model | Owned outputs / scratch |
| --- | --- | --- | --- |
| PlusState/ZeroState/CNOT | H1, CX1 | Clifford1, T0, measurement0 | 2 / 0 |
| H/CNOT unitary | H1, CX1 | Clifford2, T0, measurement0 | 2 through / 0 |

Both source peaks are two logical qubits. PlusState's preparation H is not
charged as an extra Clifford in that source model; the model remains separate
from actual primitives. No measurements occur here, so there are no sampled
branches to substitute for exact amplitudes. Existing physical binding supports
`circuit.append(op)` and caller-supplied output allocation.

## Remaining boundaries

This extends the existing Q-0885 notebook dependency evidence; it adds no new
capability checkbox or complete parent acceptance. The whole original 18-cell
notebook, its MultiAnd custom flatten instance selection, contraction-info
workflow and Quimb drawing layout remain open. No original notebook-wide
execution or layout/performance parity is claimed. The known default-suite
600-second timeout remains open.

## Local validation

19 focused cases, 1,917 related regression cases and 84 CPU/TN checks pass.
Strict typing covers both changed Python files. Black, Ruff, Simulation-team
scope, architecture, multi-level IR and protected API checks pass. The example
has 98% branch-aware coverage (80 statements, 10 branches); its script entrypoint
also runs independently. Existing source numerical warnings and two CPU/TN
warnings remain visible. The default-suite timeout was not rerun, and these
focused checks do not imply whole-repository success.
