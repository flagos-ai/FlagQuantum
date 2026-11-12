# Native tensor-simulation notebook

The executable walkthrough is
[`tensor_simulation.ipynb`](../../examples/notebooks/tensor_simulation.ipynb).
Open it in a Python 3 kernel with the repository installed and run all cells.
It produces five figures, an (8, 4) And isometry, an (128, 16) MultiAnd isometry,
and identical direct/preinitialized Bell vectors `[1, 0, 0, 1] / sqrt(2)`.
The notebook imports no Qualtran or Quimb. Existing physical component examples
retain `circuit.append(op)`; this batch adds no physical gate implementation.

## Ownership, interfaces and outputs

This is composition of existing owners, not a second tensor simulator or graph.
Compiler supplies typed register graphs, selection and zero initialization;
Simulation supplies source-ordered leaf tensors, binding and contraction;
Drawer supplies register scores and index-incidence figures. Custom CNOT uses
complex128 data with output ctrl,target followed by input ctrl,target axes.
Register arrays remain row-major/MSB-first. MultiAnd prefix/target bits and
Bell's two bits are owned logical results, not scratch promised zero at a
forward exit. Existing erasure tests check their explicit later cleanup.
The local notebook symbol callback shortens presentation labels only.

## Whole-source disposition

Reference: Qualtran v0.7.0, commit
`bcb17bb25360ed68ce3bc084645db3e929799389`,
`qualtran/simulation/tensor.ipynb`, Git blob
`d91373e7d8eb21c48a0b870d99d1bb2973971861`.
Copyright 2023 Google LLC, Apache-2.0; attribution is retained in the notebook.
All 36 cells were audited: 18 code cells and 18 Markdown cells. No saved outputs,
attachments or external image assets exist. Original Python 3 ipykernel metadata
is retained as Python 3, with native environment version 3.12 instead of 3.11.8.
The table uses zero-based source cell indices, also retained by the native file.

| Markdown cells | Code cell | Original workflow and native disposition |
| --- | --- | --- |
| 0 | 1 | CNOT dense matrix; real COPY/XOR nodes, register binding/contraction |
| 2 | 3 | PlusState/ZeroEffect; existing H-zero projection and basis covector |
| 4 | 5 | Rectangular And; source five-axis leaf in real predicate composition |
| 6, 7 | 8 | Dense alias equivalence; native tensor contraction versus physical unitary owner. Correct source prose's `(n,n)` to `(2**n,2**n)` |
| 9 | 10 | MultiAnd ladder; same three calls, native register-score view |
| 11 | 12 | Friendly-index tensor graph; native incidence view of actual leaves |
| 13 | 14 | Contraction info; existing native profile, cost-unit differences explicit |
| 15 | 16 | Internal connection inspection; integer edge plus tensor/axis endpoints |
| 17 | 18 | Partial selection; logical suppliers stop before physical decomposition |
| — | 19 | Custom `i != 2` flatten; same instances 1,2,3 and three atomic And leaves |
| 20 | 21 | MyCNOT custom data; same explicit complex128 matrix and output/input axes |
| — | 22 | Custom tensor sanity check; same consumer and matrix as factorized CNOT |
| 23 | 24 | BellState decomposition; existing Plus/Zero/CNOT typed graph |
| 25 | 26 | Automatic Bell expansion and full complex vector |
| 27 | 28 | Unexpanded Bell root refusal; normal native missing-provider diagnostic |
| 29, 30 | 31 | COPY/XOR topology, tag names and full leaf data; native drawing |
| 32 | 33 | H/CNOT input circuit; 4 by 4 unitary and register score |
| 34 | 35 | Zero initialization; same Bell vector and initialized register score |

## Reference execution and explicit presentation differences

The test verifies the original blob then executes every original code-cell string
unchanged in one namespace. Original constructors, decompositions, tensors,
flattening and contractions execute normally. Only `show_bloq` and
`TensorNetwork.draw` are intercepted at the test presentation boundary: this
machine has no Graphviz `dot`. All five presentation calls are recorded. This
is **not** a claim that the original Graphviz/Quimb renderers ran successfully.
The native notebook executes independently with the reference contraction
entry points replaced by failures, and renders all five native figures.

Native graph-owned integer indices replace foreign `(Connection, bit)` objects;
endpoint incidence preserves the corresponding structure. Native Matplotlib
incidence/register layouts replace Quimb's spectral and Graphviz coordinates.
Their pixel layouts and general Quimb optimizer interface are not transplanted.
The notebook clearly presents these differences; it does not introduce foreign
compatibility objects into production. Source numerical warnings from autoray
are reported, not suppressed or used to weaken complex-array assertions.

## Resources and acceptance

| Workflow | Measured physical primitives | Original model / numerical accounting |
| --- | --- | --- |
| Bell, direct and zero-initialized | H 1, CX 1; no measurement/T gates; two owned results, zero extra scratch | Prepared model Clifford 1; unitary model Clifford 2; peak 2 qubits |
| Four-control conjunction forward | H 6, T 6, Tdg 6, CX 18, S 3 | AND 3 / 12 T-equivalent; peak 7 qubits; three owned prefix/target outputs |
| Explicit conjunction erasure | H 3, measurement 3, CZ 3, reset 3 | Clifford 3, measurement 3; all small branches tested in conjunction regression |
| Same MultiAnd contraction path | No added physical operation | Source 9,216 opt_einsum FLOPs; native 4,608 union-dimension units; 2 steps; peak/output 2,048 elements |

The complex128 dense output occupies 32,768 bytes; this is not measured process
RAM. COPY/XOR bonds are numerical edges, not work qubits. Notebook composition
adds no physical gates, measurement branches or capability checkbox IDs.

`test_tensor_simulation_notebook.py` validates all 18 original and 18 native code
cells, nine complete complex arrays, exact COPY/XOR/And leaf data, index endpoints,
custom selection, failure behavior, profiles and five nonempty raster renders.
Related Bell and conjunction suites retain physical append/inverse, spectator,
phase/gradient, workspace and exhaustive small measurement-branch evidence.
Closing the concrete notebook asset Q-0885 does not close Q-0882/0883/0887/0889/
0891/0895, full foreign tensor interfaces, optimizer/layout equivalence or G3.
The prior default-suite 600-second timeout remains open; focused regression is
not a declaration that the entire repository passes.

Local acceptance for this batch: **1,945 related tests passed in 35.72 s**, and
**84 CPU/TN tests passed in 9.68 s**. Strict mypy covers the one new Python test
file; Black validates its formatting and every notebook code cell, and Ruff
checks both artifacts. Scope, architecture, multi-level IR and protected API
checks pass. No coverage percentage is claimed for notebook composition and no
production numerical implementation changed. Existing reference numerical
warnings remain visible. Native kernel metadata uses deterministic cell IDs;
notebook-schema validation through optional `nbformat` was unavailable locally.
