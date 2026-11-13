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
| _t_counts_for_bloq | Compiler query_terminal_t_cost traverses existing resource declarations and caches full-subtree precision-sensitive terminal costs. Visibility cutoffs are independent. Exclusive T-bearing aggregate costs require explicit calls. |
| _keep_if_small | Native default keep recognizes binary ConjunctionCall, FixedTCall and ControlledZPowerCall. Existing keep=None and max_depth retain their source visibility semantics. |
| _is_leaf_node | A node with no positive-cost successors emits one row, including a zero-cost root. A single positive T/T-adjoint child collapses through is_t_gate. |
| _populate_flame_graph_data | iter_flame_graph_data preserves successor insertion order, positive-cost filtering, DFS path titles, tab-separated cost and repeated complete subtrees. Streaming replaces list multiplication. |
| get_flame_graph_data | iter_resource_flame_graph_data composes existing graph construction, Compiler full costs, native titles and T-terminal recognition for ordered multiple roots. Streaming and caller-owned files are explicit adaptations. |
| get_flame_graph_svg_data | render_flame_graph_svg streams rows to a temporary directory and invokes an explicit executable with --countname TCounts. Input closes before execution. Cleanup, timeout and diagnostic handling are native lifecycle adaptations. Interactive browser inspection and packaged asset distribution remain open. |

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


## Native convenience and SVG workflow

Compiler `terminal_t_cost.query_terminal_t_cost` uses the existing recursive
cost-query cache and resource callee rules. Terminal costs reuse the locked
precision-sensitive model. Explicit declaration rows replace component calls
and exclusive categories. Without a row, exclusive T, Toffoli, Fredkin, And and
rotation costs require child declarations. This prevents aggregate rotation
counts from losing their synthesis precision. Symbolic queries remain available,
while folded-data rendering admits concrete nonnegative Python integers.
Raw numeric multiplicities are validated before graph aggregation can coerce
booleans. Arithmetic and other compound keys require their explicit callee rows.

```python
from flagquantum.compiler._component_trace.resource_generalization import (
    FixedTCall, PhaseCall, RotationCall,
)
from flagquantum.compiler._component_trace.resource_model import ComponentCost
from flagquantum.drawer.flame_graph import (
    iter_resource_flame_graph_data, render_flame_graph_svg,
)

root = ComponentCost("THC lookup", 3)
fredkin = FixedTCall("fredkin")
rz = RotationCall("rz", 0.17, precision=1e-4)
tdg = PhaseCall("t", adjoint=True)
calls = {root: {fredkin: 2, rz: 3, tdg: 5}}
rows = list(iter_resource_flame_graph_data(root, declared_calls=calls, max_depth=0))
assert rows == ["THC lookup(T:94)\t94"]

# Supply an installed FlameGraph script that accepts the source CLI arguments.
# svg = render_flame_graph_svg(iter(rows), executable="/path/to/flamegraph.pl")
```

Component names supply default native titles. Other resource keys use their
native string representations. A title callback controls presentation without
introducing foreign field reflection. Default visible leaves match the source's
And, Toffoli, TwoBitCSwap and CZPow class policy through native resource keys.
The renderer returns SVG bytes and callers own persistent output paths.
The executable is explicit, with no vendored dependency or asset discovery.
UTF-8 input is streamed with source-compatible inter-row newlines. Temporary
files and directories are removed on success, subprocess failure, timeout and
input-generator failure. Error diagnostics retain the final 4000 stderr bytes.
The timeout bounds subprocess execution. Input generation remains caller-owned.

### Acceptance and retained boundaries

There are 45 new workflow cases and eight SVG/lifecycle cases. Four untouched
source examples, CSwap, MultiAnd, PrepareUniformSuperposition and QFTTextBook,
match ordered rows under both keep policies and four depth settings. Tests
translate source identities and declarations at the reference boundary. Full
T costs are computed by the native Compiler query. No source cost map is copied.
Shared DAGs, caller caches, multiple roots, wrapped counts, symbolic costs,
exclusive-cost rejection and cycles hidden by display cutoffs are covered.

The actual native CSwap SVG equals untouched source get_flame_graph_svg_data
output after sorting the children of the SVG frames group. The Perl renderer
uses hash iteration, so nonoverlapping frame element order varies across runs.
No geometry, color, title, count or JavaScript is normalized. The original
source module's asset lookup is redirected to the verified script only at the
test boundary. Script, license and metadata match frozen blobs
8c917ecf323e13c1ab3c1107c1963f33dec0123e,
2230598ebad9e470736e12119db740007c1a764b and
8c6696d1a9b9c36c929d0a01a00dfe46f4de1cbb respectively.
They remain reference-environment assets. The source code is unchanged.

Local acceptance reports 53 focused cases, 961 concentrated cases and 86
independent integration cases, with zero skips. Strict typing covers four changed
Python files. Black, Ruff, Compiler scope, architecture, internal IR and protected
API checks pass. The complete Drawer flame owner covers 73 statements and 24
branches at 100 percent. Together with the existing terminal-cost owner,
coverage is 98 percent over 125 statements and 54 branches. Two uncovered
statements belong to existing terminal error paths. A native 94-T SVG example
executes successfully. Evidence uses the g2-flame-composed- and
g2-flame-workflow- prefixes in the external reference environment.

The in-app browser rejects file-protocol preview, so interactive zoom/search
inspection remains open. XML comparison includes their original script bodies.
Foreign name reflection, executable distribution, historical QFT T190 versus
T496 assertions and general foreign/serialized-symbol admission remain explicit
gaps. Q-0739 and Q-0740 keep their parent status open. Actual Fredkin emission
remains CX8/H2/T4/Tdg3, zero work and zero measurements. Terminal-model Fredkin7T
and QEC default Fredkin4T remain separately reported. The native mixture model
is 2*7 + 3*25 + 5 = 94T. Drawing adds zero gates, work qubits or measurements.
Existing complex-amplitude, measurement-branch, work-zero and circuit.append
regressions pass. The historical full-default 600-second timeout stays open.
