# Resource counting package entry source-file audit

Q-0797 covers all 21 unique imports in Qualtran v0.7.0
`resource_counting/__init__.py` at
`bcb17bb25360ed68ce3bc084645db3e929799389`.
The initializer imports `GeneralizerT` twice and binds the same type alias.

| Source name | Native owner | Native responsibility |
| --- | --- | --- |
| `GeneralizerT` | `flagquantum.compiler._component_trace.call_graph:Generalizer` | Ordered typed resource rewrites and omission. |
| `BloqCountDictT` | `flagquantum.compiler._component_trace.call_graph:RawCallees` | Symbolic count mappings at the callee boundary. |
| `BloqCountT` | `flagquantum.compiler._component_trace.call_graph:RawCallees` | Weighted callee pairs in a declaration. |
| `big_O` | `flagquantum.compiler._component_trace.symbolic_math:asymptotic_order` | Symbolic order at infinity. |
| `MutableBloqCountDictT` | `flagquantum.compiler._component_trace.call_graph:RawCallees` | Mutable builders can supply mappings to the read-only boundary. |
| `SympySymbolAllocator` | `flagquantum.compiler._component_trace.call_graph:SymbolAllocator` | Query-local symbolic placeholders. |
| `get_bloq_callee_counts` | `flagquantum.compiler._component_trace.call_graph:component_callee_counts` | Normalize, rewrite and combine direct callees. |
| `get_bloq_call_graph` | `flagquantum.compiler._component_trace.call_graph:symbolic_call_graph` | Build weighted declarations and terminal totals. |
| `build_cbloq_call_graph` | `flagquantum.compiler._component_trace.program_call_graph:program_component_call_graph` | Recover calls and direct multiplicities from a verified typed Program. |
| `format_call_graph_debug_text` | `flagquantum.drawer.call_graph:format_call_graph_debug_text` | Render a stable textual view of the graph. |
| `get_cost_value` | `flagquantum.compiler._component_trace.cost_query:query_component_cost` | Query one metric with seed, static value and recursion precedence. |
| `get_cost_cache` | `flagquantum.compiler._component_trace.cost_query:CostMetric.cost_cache` | Collect or extend a metric cache. |
| `query_costs` | `flagquantum.compiler._component_trace.cost_query:query_component_costs` | Merge independent metric rows. |
| `CostKey` | `flagquantum.compiler._component_trace.cost_query:CostMetric` | Define a recursive metric and its identity and static validator. |
| `CostValT` | `flagquantum.compiler._component_trace.cost_query:Value` | Carry the cost type independently of the component key. |
| `SuccessProb` | `flagquantum.compiler._component_trace.cost_query:SUCCESS_PROBABILITY` | Model concrete success probabilities, with a separate symbolic declaration entry. |
| `QubitCount` | `flagquantum.compiler._component_trace.program_logical_width:query_program_logical_width` | Model live logical width of a verified Program, with a separate declared fallback. |
| `BloqCount` | `flagquantum.compiler._component_trace.selected_call_count:count_selected_calls` | Count selected typed gates at the selected graph boundary. |
| `QECGatesCost` | `flagquantum.compiler._component_trace.qec_gate_cost:query_qec_gate_cost` | Query categorized native or historical QEC model costs. |
| `GateCounts` | `flagquantum.compiler._component_trace.qec_gate_cost:QECGateCost` | Store and combine symbolic and concrete gate categories. |
| `generalizers` | `flagquantum.compiler._component_trace.resource_generalization` | Own modeled identity rewrites. Ecosystem owns SDK gate conversion. |

The initializer's responsibilities are distributed among these native owners.
`flagquantum.compiler.estimate_resources` reports static instructions after
normal append. `resource_model_report` separately communicates declared model
cost, optional emitted instruction counts and execution evidence. The audit
tests resolve every mapped name and compare the same exact Fredkin operation
through the public circuit and declared resource paths. Individual file
audits retain their documented boundaries.
