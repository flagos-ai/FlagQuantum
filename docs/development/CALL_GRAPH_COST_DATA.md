# Native call-graph cost-table formatting

## Ownership and interface

Drawer owns `format_call_graph_cost_data` in `flagquantum/drawer/call_graph.py`.
Compiler owns the existing `GATE_COST`, `PEAK_LOGICAL_QUBITS`, recursive cost
queries, `GateCost`, `QECGateCost` and aggregation budgets. The formatter consumes
caller-owned metric tables and returns fresh string annotations. It preserves
node, metric and rendered-key insertion order, including colliding rendered
keys. Unknown metrics retain their string key/value and emit `UserWarning` at
the caller. Invalid gate values raise `TypeError` before returning a result.
No graph, cache, circuit, register or numerical state is mutated.

The existing `export_modeled_call_graph_dot` entry queries both metrics through
`query_component_costs` and calls the same formatter. Its depth limit affects
display while costs retain the full declared tree. Explicit symbolic gate and
width rows use the same formatter. Symbolic width inference is a separate scope.

```python
from sympy import Symbol
from flagquantum.compiler._component_trace.cost_query import GATE_COST, PEAK_LOGICAL_QUBITS
from flagquantum.compiler._component_trace.qec_gate_cost import QECGateCost
from flagquantum.drawer.call_graph import format_call_graph_cost_data

annotations = format_call_graph_cost_data(
    {"root": {PEAK_LOGICAL_QUBITS: Symbol("n"), GATE_COST: QECGateCost(t=Symbol("t"))}}
)
assert annotations == {"root": {"Qubits": "n", "Ts": "t"}}
```

## Locked source definition disposition

Qualtran v0.7.0 commit `bcb17bb25360ed68ce3bc084645db3e929799389` remains
unchanged. Focused tests verify Git blobs for `drawing/bloq_counts_graph.py`
and `resource_counting/_bloq_counts.py` against the frozen inventory.
Apache-2.0 copyright and source attribution remain in the production owner.

| Source definitions | Native owner and disposition |
| --- | --- |
| `_CallGraphDrawerBase.__init__`, `get_id`, `add_nodes`, `add_edges`, `get_graph` | `_call_order`, `export_call_graph_dot`, deterministic IDs and existing edge multiplicity formatting |
| Base `get_node_title`, `get_node_details`, `get_node_properties` | Existing title callback and annotation table. Arbitrary subclass property overrides remain open |
| Base `get_svg_bytes`, `get_svg` | Existing bounded Graphviz renderer and notebook SVG presentation |
| `GraphvizCallGraph.__init__`, `get_node_title`, `get_node_properties` | Existing explicit graph and annotation entry. Native HTML table presentation is an adaptation |
| `format_qubit_count` | Existing peak metric becomes the `Qubits` string row |
| `format_qec_gates_cost` | `_gate_annotations`, existing shared T/CCZ, Beverland and legacy budget owners. Concrete and symbolic values supported |
| `format_cost_data` | `format_call_graph_cost_data`, added in this batch |
| `from_bloq` | Existing concrete modeled exporter now uses the shared formatter. General typed-resource factory/foreign admission remains open |
| `_format_bloq_expr_markdown`, `format_counts_graph_markdown`, `format_counts_sigma` | Existing Markdown exporters and source assertion replacement tests |

Native metric identity uses existing Compiler constants. Source key class
instances stay at the test reference boundary. The native invalid-value error
is `TypeError`, where the source uses an assertion. Native unknown-key warning
says "metric" where source says "key". All five gate aggregation outputs,
including insertion order and symbolic expressions, match the original helper.
The default total-T budget retains four per Toffoli, controlled swap and AND,
and eleven per rotation. These values describe the declared model.

## Verification and resources

The 62 new cases cover 40 complete concrete/symbolic/order combinations, unknown
metric collisions and warnings, invalid values with retry and input ownership,
invalid aggregation dispatch, 15 full-cost depth/aggregation cases through the
existing exporter, escaped annotation values and actual SVG parsing.

Focused tests pass 62 cases in 1.90 seconds. Concentrated regression passes
1310 cases in 28.50 seconds with 1112 existing source/dependency warnings.
Independent real-kernel notebook/CPU/TN integration passes 87 cases in 12.12
seconds with two existing warnings. All runs have zero skips. The locked
Qualtran, quimb and native Kickmix Simulator are verified before regression.
Strict typing covers both changed Python files. Black, Ruff, Compiler scope,
architecture, internal IR and protected API checks pass.

Measured coverage is 100 percent for the existing call-graph example,
94 percent for Drawer call_graph and 95 percent combined. Uncovered lines
belong to existing Markdown edge-label and renderer failure paths.
An initial strict check exposed invariant mapping-key annotations. A generic
metric-key annotation now accepts existing typed Compiler tables without casts.
A file-path doctest invocation also failed on relative imports. Importing the
module before doctest passes all three examples, and the Markdown example
executes successfully. Both diagnostic logs are retained alongside the passing
results.

| Accounting category | Observed or declared value |
| --- | --- |
| Formatting added primitive gates / work wires / measurements | 0 / 0 / 0 |
| Synthetic source atom and root model | 100T and 100*n + 600T |
| Synthetic atom numerical action | X |
| Existing exact Fredkin emitted primitives | CX8, H2, T4, Tdg3, work0, measurement0 |
| Exact Fredkin terminal-T / default QEC estimate | 7T / 4T |
| Existing measured merged-Fredkin emitted T-equivalent / measurement | 11 / 1 |
| Merged model components | 4 + 7 + 0 |

Existing concentrated tests retain gate-sequence, complex-amplitude, measurement
branch, work-wire cleanup, exact adjoint and `circuit.append(op)` validation.
This formatter consumes costs without executing a quantum operation.

Q-0734.03 records this capability. Q-0734 remains open for generic factory and
property-override scope. Foreign admission, CDDL renderer packaging, browser
interaction, the historical QFT 190 versus 496 expectation and the historical
full-default 600-second timeout retain their independent open records.
