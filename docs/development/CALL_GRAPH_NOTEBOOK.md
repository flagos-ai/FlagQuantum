# Resource call-graph notebook

The complete locked `drawing/drawing_call_graph.ipynb` asset is represented by
`examples/notebooks/resource_call_graphs.ipynb` and the runnable native example
`examples/algorithms/resource_call_graphs.py`. Compiler owns declared dependencies,
symbolic traversal and recursive QEC costs. Drawer owns title and annotation
formatting, DOT export and bounded SVG rendering. This presentation workflow is
a single_device_fast_path. It constructs no register IR or execution outputs.
There is no bit order or quantum work-wire ownership in the declaration graph.

## Run the native workflow

```bash
python -m examples.algorithms.resource_call_graphs
python -m pytest tests/unit/test_call_graph_notebook.py -q
python -m pytest tests/integration/test_register_graph_notebook.py -q
```

Graphviz must be installed. The example prints the declared root cost
`100*_n0 + 600` and a rendered SVG byte count. Open the notebook for the plain
and annotated graph displays. The existing title callback and annotation mapping
supply customization. Resource node keys preserve identity independently of text.
The example owns its declarations. Production requires no reference SDK,
NetworkX, pydot or new graph representation.

## Whole-asset disposition

The reference is Qualtran v0.7.0 at
`bcb17bb25360ed68ce3bc084645db3e929799389`. The original notebook Git blob is
`8e8c368bbb9400d76d95d183070efe66f7c5a841`. A missing local reference copy was
restored from that exact blob in the existing source checkout. Its content was
verified before use. The five source assets used by the tests are checked
against the frozen inventory.

| Original cell | Native disposition |
| --- | --- |
| 0, call-graph introduction | Native graph, cost and rendering ownership described |
| 1, declaration and convenience display | Explicit native declarations, symbolic_call_graph, DOT export and SVG display |
| 2, Graphviz drawer explanation | Existing native DOT and renderer interfaces explained |
| 3, first 400 SVG bytes | Full SVG bytes rendered, retained and prefix printed |
| 4, cost annotation explanation | Recursive native cost query and synthetic model convention described |
| 5, full per-node cost table and SVG | Existing QEC cache supplies every node's T, Clifford and rotation fields |
| 6, customization hooks | Existing title callback and annotation mapping described |

The native notebook retains seven cells, comprising three code cells and four
Markdown cells, with cleared outputs and unique IDs. All original code cells run
without changing their source. Source display hooks are captured only at the test
boundary to retain all three rendered SVGs. The native path separately runs all
code with imports of Qualtran, Cirq, Quimb, NetworkX and pydot rejected. The real
Jupyter kernel integration checks its three execution counts and two displayed
SVG outputs. The third SVG is exported and printed as bytes in both notebooks.

Node names adopt native declaration names. Native DOT tables retain the existing
Drawer layout. Source HTML font wrappers and cell padding are presentation
adaptations, so pixel equality is outside this asset's acceptance. Tests compare
all ordered direct edges and leaf counts at depths 0, 1, 2, 3 and unlimited,
all nine per-node cost tables, table rows, edge labels and graph orientation.
Symbol substitution includes zero and large multiplicities. Custom title and
annotation escaping and the standalone normal entry point are covered. The
rendered native annotated graph was inspected locally.

## Resource accounting and execution evidence

| Layer | Verified convention |
| --- | --- |
| Synthetic atomic declaration | 100 T, zero Clifford and rotation counts |
| Parallel and serial model | 300 T each |
| Root model | 100*n + 600 T |
| Drawing overhead | Zero primitive gates, work wires or measurements |
| Physical circuit count for this synthetic graph | No circuit emitted |

These declarations follow the source demonstration's static 100T atomic cost.
The source atom's numerical tensor is X. Its static resource cost is independently
declared, so it supplies no physical 100T execution witness. Existing exact
Fredkin complex-amplitude and measured merged-Fredkin branch/work-zero/append
regressions remain in the concentrated suite. Their physical instruction scans
and terminal or QEC model estimates stay separate in RESOURCE_FLAME_GRAPH.md.

Initial validation corrected a test assumption that source tables had no outer
font element, explicit types for cost caches and pydot dynamic attributes, and a
Ruff dict-comprehension diagnostic. The independent integration launcher's
missing multiprocessing main guard caused a cotengra trial bootstrap failure.
The temporary launcher was repaired and the full integration rerun. Initial
logs are retained alongside accepted evidence under g2-call-notebook-.

## Batch acceptance

Implementation 303b4c33 passes 13 focused cases in 1.80 seconds, 1225 concentrated
cases in 24.79 seconds and 87 independent notebook/CPU/TN integration cases in
10.91 seconds, with zero skips. The new notebook adds one real-kernel case.
Focused and concentrated runs retain eight and 1112 dependency/source warnings.
Integration retains two existing warnings. Strict typing of three files, Black,
Ruff, Compiler scope, architecture, IR and protected API checks pass.
The new example covers all 31 statements and two branches. Existing call-graph
Drawer coverage is 92 percent, including its preexisting uncovered Markdown,
invalid-aggregation and renderer error paths. Combined measured coverage is
93 percent. This is the measured scope of this batch.

Q-0738 is accepted as a complete asset with child Q-0738.01. The source graph
formatter and generic display facade retain their separate parent items.
Evidence includes the five frozen source blobs, retained source/native SVGs,
local PNG inspection, complete original/native cell execution and the standalone
normal entry point. The primary ledger and stage overview are synchronized.
