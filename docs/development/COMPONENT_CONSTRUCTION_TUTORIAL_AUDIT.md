# Complete component construction tutorial

Q-0155 maps all 42 cells of the unchanged source
`qualtran/_infra/composite_bloq.ipynb` at v0.7.0 commit
`bcb17bb25360ed68ce3bc084645db3e929799389`. Its Git blob is
`1fe7ddcab53ddd9a909417c8994d643d8f9d8ff4`. The source notebook executes in
the reference environment. The native notebook is
`examples/notebooks/component_construction.ipynb`, with each cell carrying
`frozen_source_cell` metadata. Source copyright and license are retained.

## Ownership and cell coverage

| Source cells | Native behavior and owner |
| --- | --- |
| 0–5 | Explicit Program construction, straight and crossed CNOT calls, ordered return binding. Core owns the graph, Compiler owns the builder. |
| 6–9 | Explicit operation decomposition and executable native CNOT definition. Drawer renders the graph. |
| 10–17 | Named predecessor/successor iteration, complete Core verification and directed connection inspection. |
| 18–23 | Existing immutable graph copy and explicit presentation offsets for occurrence labels. |
| 24–25 | Manual copy through ordered traversal, scalar/array replacement pairs and ordinary live builder calls. |
| 26–30 | Nested calls and construction-time inlining in the first, second and middle position. |
| 31–37 | One-child wrapper, one-level expansion, retained decomposed child bodies and actual splicing. |
| 38–41 | Source occurrence-label predicate, repeated expansion, bound Circuit.append execution and separate primitive/model reports. |

`component_construction_tutorial.py` only composes existing domain owners.
The builder prerequisite is documented in `COMPONENT_INLINING_AUDIT.md`.
No foreign program object enters production execution. The native notebook
executes with imports from the three upstream quantum packages refused.

## Exactness and explicit adaptations

Source and native immediate-call connections match at every constructed stage,
including ordered input ports, scalar/array shape, predecessor register names,
element indices and producer identity. Comparisons normalize occurrence labels
only when comparing topology; selective expansion uses the actual retained
source label 1. After the wrapper is expanded, that label denotes the first
child. Source and native tensor maps agree exactly, including the crossed-CNOT
return-wire permutation. Full complex inputs include an untouched spectator.
No measurement branch exists in these examples. Bound workspace event lists are
empty, so they introduce no work qubits requiring cleanup.

The copy demonstration uses explicit presentation offsets instead of changing
the production builder constructor. Native Program definitions already carry
their decomposition bodies. The source's conversion of a child into a nested
CompositeBloq is represented by a separately named definition with the same
body. Its call boundaries and tensor behavior remain unchanged until expanded.
The tutorial's manual copy includes an explicit input mapping instead of relying
on equality between two builders' left-boundary handles.

## Resource accounting

| Example | Measured primitives | Source model |
| --- | --- | --- |
| Straight/crossed two-CNOT program | 2 CX | 2 Clifford CNOTs |
| Single parallel component or its copy | 3 X | Synthetic 300 T |
| Two nested or selectively inlined parallel components | 6 X | Synthetic 600 T |
| Three parallel components at every expansion stage | 9 X | Synthetic 900 T |

The source TestAtom tensor is X while its declared static cost is deliberately
100 T. That source model is retained as a model, independently of measured
native primitives. The full tutorial verifies the 900-T model against the
unchanged source cost evaluator. Graphviz renders native directed graph SVGs.
Existing inlining tests separately cover zero-width/shaped inputs, directional
allocation/free, duplicate-consumption failure prefixes and definition clashes.
The source empty-output defect remains recorded in the prerequisite audit.

## Validation evidence

The concentrated tutorial, inlining, builder, traversal and rebinding regression
passes 130 cases without skips. Strict typing covers the five changed Python
files with explicit package bases. Black formats Python and every notebook code
cell, Ruff passes, and Compiler scope and architecture checks pass. Graphviz SVG
generation is exercised for crossed controls, copying, selective construction
and full expansion. The full dependency check retains 706 existing findings,
with zero findings in this batch's paths. It is not a repository-wide green
dependency check.
