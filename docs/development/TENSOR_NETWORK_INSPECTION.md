# Native tensor-network topology inspection

Drawer consumes existing `TensorNetworkNode` objects and the existing register
index-name side table. It reads shapes and integer labels only; Simulation keeps
ownership of data, index allocation, contraction and gradients. This is a local
read-only presentation path, with no new Core schema or stable export.

```python
from examples.algorithms.tensor_network_inspection import inspection_example
from flagquantum.drawer.tensor_network import (
    draw_tensor_network, export_tensor_network_dot, tensor_index_occurrences,
)

nodes, names = inspection_example()
print(tensor_index_occurrences(nodes))
source = export_tensor_network_dot(nodes, index_names=names)
figure, axes = draw_tensor_network(nodes, index_names=names)
figure.savefig("controlled-not.png")
# The caller owns and closes the figure.
```

The example constructs a real native CNOT component graph and supplies the locked
COPY/XOR leaves. A single-occurrence label is outer; repeated labels are inner.
Each incidence records a tensor ordinal and an axis. Tensor and index identities
are independent of display names. Repeated trace axes, multi-tensor hyperedges,
scalars and zero-sized dimensions are retained. Existing Simulation validation
rejects rank/label mismatches and inconsistent shared dimensions; Drawer rejects
negative/non-integer labels and names for absent indices.

DOT export needs no Graphviz installation. Matplotlib is optional and imported
only when drawing; neither path reads tensor values, transfers devices, allocates
work qubits, chooses contraction paths or contracts the network. Meta tensors
therefore support both presentations. Layout is deterministic and finite: it
provides incidence inspection, not Quimb spectral coordinates, pixel equivalence,
contraction-info reports or large-network readability guarantees.

## Locked-source evidence

Qualtran v0.7.0, commit `bcb17bb25360ed68ce3bc084645db3e929799389`,
`qualtran/simulation/tensor.ipynb` uses `cbloq_to_quimb`, friendly indices,
inner-index inspection and `TensorNetwork.draw`. The notebook blob is
`d91373e7d8eb21c48a0b870d99d1bb2973971861`. The native view implements that
inspection dependency through FlagQuantum-owned data and presentation.

Tests compare every tensor-axis incidence against unmodified
`cbloq_to_quimb` for 1, 2, 3 and 7 CNOTs, with structured and friendly indices.
Complete complex matrices agree before/after inspection and against the source.
Trace, hyperedge, scalar, empty, escaped/colliding names, meta tensors, autograd
and executable DOT/PNG output exercise the presentation boundaries. Rendering
retains the original node and tensor identities.

| Repeated CNOTs | Source/native nodes | Source/native axis incidences | Physical CX | Source modeled Clifford |
| --- | --- | --- | --- | --- |
| 1 | 2 | 6 | 1 | 1 |
| 2 | 4 | 12 | 2 | 2 |
| 3 | 6 | 18 | 3 | 3 |
| 7 | 14 | 42 | 7 | 7 |

Drawing adds zero physical primitives, work qubits, measurements or feedback
slots. Physical counts and source QEC estimates above reuse the separately
verified factored CNOT example; topology metadata is not a new gate count.

This accepts one bounded notebook inspection dependency. The complete original
18-cell tensor notebook remains open, including And/MultiAnd/Bell/custom-provider
composition, flattening and contraction-info workflows. No parent source file or
performance claim closes here. The existing default-suite timeout remains open.

## Local validation

The focused source/topology/rendering suite passes 24 cases. Concentrated related
regression passes 1,822 cases; CPU/TN checks pass 84; existing Drawer contract
checks pass 51. Strict typing covers the three changed Python files. Black,
Ruff, compiler-team scope, architecture, multi-level IR and protected API checks
pass. New Drawer coverage is 100%; combined Drawer/example coverage is 98%
(97 statements, 26 branches). Existing reference numeric overflow warnings and
two CPU/TN warnings remain; no default-suite success is inferred. The generated
CNOT PNG was visually reviewed, including distinct axis labels.
