# Reflection notebook source closure

Q-0665 is the complete Qualtran v0.7.0 `reflections/reflections.ipynb` at
`bcb17bb25360ed68ce3bc084645db3e929799389`, frozen blob
`e4d3316f906b697634149f4e61903333d947e7af`. The native artifact is
`examples/notebooks/prepared_state_reflections.ipynb`. Its opening cell retains
Google LLC 2023 Apache-2.0 attribution.

## Complete cell mapping

The first nineteen native cells preserve the original code/Markdown sequence.
Two additional cells demonstrate native append and independent resource accounts.

| Original cells | Native responsibility |
|---|---|
| 0–3 | Definition, scalar/control semantics and owned imports. The mathematical inverse and original source-declared adjoint defect remain explicit. |
| 4–6 | Five equal alias weights at precision .01, and zero reflection over 1/2/3 bits with control one and coefficient -1. Reuse accepted native declarations without alternate arithmetic. |
| 7–8 | Two graphical signatures through existing Drawer signature and Graphviz owners, retaining ordered widths/directions. |
| 9–10 | One-level reflection call graph and leaf counts, including distinct PREPARE directions, three-bit zero-controlled Z and two X calls. Query independent native resource model. |
| 11–14 | Identity PREPARE description and complete 10/4/1 example using accepted native metadata/binding. |
| 15–16 | Identity graphical signature through the same Drawer owner. |
| 17–18 | One-level Identity(10), Identity(4), Identity(1) graph and leaf counts. |
| Added 19–20 | Ordinary `Circuit.append(op)`, full complex state and clean borrowed anchor, emitted primitive inventory and separate declaration estimate. |

All original code cells execute unchanged in the reference environment, and the
original notebook pytest passes in an isolated kernel. Native execution rejects
all Qualtran/Cirq imports. Original and native signatures, alias normalization,
resource declarations and one-level graphs are compared directly. The test
executes every native cell and checks all ten display objects and five SVG
outputs. The native diagrams are rendered to PNG for visual inspection, with
legible register tables and complete call edges. The shared Drawer supplies the
native layout.

## Runnable workflow and resource accounts

Open the native notebook and run all cells. Its final example is equivalent to

```python
import torch
from flagquantum import Circuit
from flagquantum.algorithms.primitives.prepared_state_reflection import PreparedStateReflection

reflection = PreparedStateReflection.around_zero((1,), global_phase=-1)
op = reflection.make_on(selection={"reg0_": (0,)}, reflection_workspace=(1,))
circuit = Circuit(2, dtype=torch.complex128).h(0)
circuit.append(op)
# Full amplitudes are (1, 0, -1, 0)/sqrt(2). Borrowed wire 1 is zero.
```

The emitted example contains two explicit one-wire Identity instructions, four X
instructions, one CZ and one scalar phase instruction. There are zero measurements
and one unitary branch. The separately queried original declaration has five
Clifford gates, zero modeled T equivalents and peak two logical qubits. The two
Identity directions collapse to one dictionary key in the original graph. Its
QEC contribution is zero. These accounts are tested independently.

The original five-state alias-reflection simulations remain original xfails, as
recorded in `PREPARED_STATE_REFLECTION_FILE_AUDIT.md`. The large 10/4/1 identity
example is declaration and drawing evidence, with physical Identity behavior
already covered by the accepted complete Identity module. This notebook introduces
no new physical admission or approximation.

The batch regression passes 283 cases without skips or warnings. Strict mypy
checks the test and all exact extracted notebook code. Black/Ruff, Algorithms
scope, architecture, multilevel IR and protected API checks pass.
