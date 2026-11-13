# Resource flame graph data

Drawer owns folded-stack formatting in `flame_graph.iter_flame_graph_data`.
Compiler retains graph construction, terminal identities and cost evaluation.
The formatter consumes the existing hashable node keys, ordered direct-call map
and full-subtree T-cost map. It introduces no graph representation or dependency.
This is a single_device_fast_path presentation workflow.

## Executable native workflow

```python
from flagquantum.compiler._component_trace.call_graph import modeled_call_graph
from flagquantum.compiler._component_trace.cost_query import GATE_COST, query_component_cost
from flagquantum.compiler._component_trace.resource_model import ComponentCost, CostCall, GateCost
from flagquantum.drawer.flame_graph import iter_flame_graph_data

leaf = ComponentCost("And", 3, GateCost(and_bloq=1))
branch = ComponentCost("Branch", 3, calls=(CostCall(leaf, 2),))
root = ComponentCost("Root", 3, calls=(CostCall(branch, 3), CostCall(leaf, 5)))
costs = {}
query_component_cost(root, GATE_COST, cache=costs)
totals = {node: cost.total_t_count(ts_per_cswap=7) for node, cost in costs.items()}
visible = modeled_call_graph(root, max_depth=1)
rows = list(iter_flame_graph_data(
    root, visible.direct_calls, totals, title=lambda node: node.name,
))
assert rows == ["Root(T:44);Branch(T:8)\t8"] * 3 + ["Root(T:44);And(T:4)\t4"] * 5
assert sum(int(row.split("\t")[1]) for row in rows) == 44
```

For large exports, iterate directly and write each row to a caller-owned stream.
The generator retains the graph and current path independently of emitted row
count. Each row excludes a trailing newline. Callers own file creation and
newline policy. Existing `estimate_terminal_t_cost` supplies precision-sensitive
costs for resource-key graphs. The test uses native Fredkin, Rz with precision
1e-4 and T-adjoint keys to obtain 7, 25 and 1 per call, totaling 94.
The title callback supplies stable native names. `is_t_gate` explicitly identifies
T and T-adjoint terminal keys when their single-child frames should collapse.

## Locked source call chain

Reference version is Qualtran v0.7.0 at
`bcb17bb25360ed68ce3bc084645db3e929799389`. The immutable inventory verifies
these four source blobs before comparisons.

| Source | Git blob |
| --- | --- |
| drawing/flame_graph.py | 33e09396e60d1775d7da5d825a6bed040a5ab55e |
| drawing/flame_graph_test.py | b5dde880fc545b244cb3bd3f15fb648d5c50239d |
| resource_counting/_call_graph.py | 8c1a6debd8c8a2e06d3a082dc6b6cd6656e4f522 |
| resource_counting/t_counts_from_sigma.py | eb4578e362849eb3e7719944bb7c01b2dc9adbc4 |

| Source definition | Native ownership and current disposition |
| --- | --- |
| _pretty_arg | Stable caller-supplied title replaces foreign field reflection. Source-specific tuple, ndarray and float display remains pending. |
| _pretty_name | Native names arrive through the existing title callback convention. Tests use source names at the reference boundary for exact row comparisons. Automatic foreign reflection remains pending. |
| _t_counts_for_bloq | Compiler query_component_cost, symbolic_call_graph and estimate_terminal_t_cost own full costs. Caller-supplied totals are independent of visible depth cutoffs. Source global graph-keyed cache is replaced by caller-owned cost maps. |
| _keep_if_small | Existing graph-builder keep and max_depth controls select visible leaves. Source automatic class recognition remains pending. |
| _is_leaf_node | A node with no positive-cost successors emits one row, including a zero-cost root. A single positive T/T-adjoint child collapses through is_t_gate. |
| _populate_flame_graph_data | iter_flame_graph_data preserves successor insertion order, positive-cost filtering, DFS path titles, tab-separated cost and repeated complete subtrees. Streaming replaces list multiplication. |
| get_flame_graph_data | Existing graph construction plus the iterator covers one root. Callers compose multiple roots and own file output. The source convenience facade remains pending. |
| get_flame_graph_svg_data | Perl execution, temporary-file lifecycle and interactive SVG remain pending under Q-0739. |

Validation completes before the first yielded row. Native admission requires
concrete nonnegative Python integers for multiplicities and costs, with an entry
for every visible node. Existing topological validation detects cycles. Titles
containing semicolon, tab or line-break separators are rejected. These explicit
portable-format boundaries are native adaptations. Source helpers accept broader
Python values and can produce ambiguous frame text.

The formatter follows source behavior when full costs and visible edge weights
differ. The source's parent/child sum assertion is disabled. Both T-child collapse
and an ordinary child have direct mismatch comparisons. A parent labeled T14 can
therefore yield a child row weighted 7 when the supplied graphs disagree. Cost
map provenance and any such mismatch must remain visible in reports.

## Verification and resource accounting

The new test file covers CSwap widths 1, 2, 3 and 10 across four visible depths,
MultiAnd sizes 3, 4, 6 and 12 in both directions with full and retained leaves,
three state preparations containing precision-sensitive rotations, shared DAG
paths, native component and terminal-cost owners, trillion-repetition partial
consumption, invalid input and retry. Exact comparisons retain row order.
Reference graphs are translated to integer keys only in tests. Source metadata
fixtures exercise the supplied-graph helper without claiming a decomposition of
those fixture graphs.

All four original test functions retain their assertions. CSwap, MultiAnd and
uniform preparation run through source and native producers. The QFT data agrees
with actual source output, while the original QFT assertion fails for both
producers. The original file independently reports three passes and one xfail.
Its known expectation remains recorded by the source issue 1474. Reference
imports initially required Protobuf, restored as 5.29.5 in the isolated reference
environment. The initial MultiAnd fixture used an unsupported constructor keyword
and was corrected to call adjoint(). Missing type annotations and an incorrect
regression filename were corrected before final acceptance. Logs retain these
failures under g2-flame-data-.

The exact Fredkin execution regression checks all eight basis columns, complete
complex amplitudes with an entangled spectator, involution and ordinary
circuit.append(op). Existing measured MultiAnd/adjoint and clean-work regressions
continue through their owners. Folded-stack generation adds zero primitive gates,
work wires or measurement branches.

| Accounting layer | Verified values |
| --- | --- |
| Actual exact Fredkin primitives per bit | CX8, H2, T4, Tdg3, work0, measurement0 |
| Historical terminal T model per Fredkin | 7 |
| QEC model default T-equivalent per Fredkin | 4 |
| Native precision-sensitive mixture | Fredkin2 at 7, Rz3 at 25, T-adjoint5 at 1, total94 |
| Component display example | Full declared T44 with depth-one visible leaves |

The folded T costs are model estimates. Physical instruction scans and model
conventions retain separate accounting. Exact local run counts and timing are
recorded in the batch acceptance section below. Q-0739 and Q-0740 remain open for
the pending source facade, reflection, rendering and QFT expectation disposition.
The Q-0747 foreign graph and serialized-symbol boundaries retain their own work.

## Batch acceptance

Implementation `b7111c18` passes 69 focused cases in 1.80 seconds, 857 concentrated
cases in 28.79 seconds, 51 model/cost/Fredkin cases in 9.53 seconds and 86
independent kernel/CPU/TN integration cases in 35.67 seconds. These runs have
zero skips. Focused and concentrated runs expose 370 source deprecation warnings
and 378 total warnings respectively. Integration retains two existing warnings.
The original test file independently reports three passes and one xfail in
0.95 seconds with 38 source deprecation warnings. Running its QFT assertion with
xfail disabled confirms an actual T190 root against its historical T496
expectation. Native actual QFT data agrees line-for-line with source and retains
the same original assertion failure.

Strict typing of the two changed Python files, Black, Ruff, Compiler scope,
architecture, internal IR and protected API checks pass. The new owner covers
all 28 statements and 16 branches. The documented native example runs with
modeled T44. Evidence uses g2-flame-data-. The primary ledger records the accepted
slices as Q-0739.01 and Q-0740.01, while both parents retain their remaining scope.
Counts are 30 accepted parents, 1250 open, 388 active and 1189 capability records.
