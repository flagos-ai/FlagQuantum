# Call-count graph drawing source audit

This audit covers the complete `qualtran/drawing/bloq_counts_graph.py` file
from Qualtran v0.7.0 at `bcb17bb25360ed68ce3bc084645db3e929799389`.
The frozen Git blob is `bfbeea4cab610adaa848cb5df0e94cbd42f12c74`.
Drawer owns presentation and Compiler owns resource identities, recursive cost
queries and logical width. The production Drawer carries the source attribution.

## Definition map

| Source definition | Native execution |
| --- | --- |
| Base `__init__`, `get_id` | Export-local graph and stable ID map in `export_call_graph_dot` |
| Base `get_node_title`, `get_node_details` | Title and details callbacks |
| Base `get_node_properties` | Rectangular title/details label and complete property callback |
| Base `add_nodes`, `add_edges` | Topological node traversal, symbolic edge labels and DOT serialization |
| Base `get_graph` | DOT text from `export_call_graph_dot` |
| Base `get_svg_bytes`, `get_svg` | Bounded Graphviz rendering in `render_call_graph_svg` and notebook display of the resulting SVG |
| Graphviz `__init__` | Explicit caller-owned annotations |
| Graphviz `format_qubit_count`, `format_qec_gates_cost`, `format_cost_data` | Shared metric formatter, including sparse rows, five gate aggregations and unknown-key warnings |
| Graphviz `from_bloq` | Typed and concrete Drawer factories with full cost queries and independent display depth |
| Graphviz `get_node_title`, `get_node_properties` | String titles, cost tables and per-node override callback |
| `_format_bloq_expr_markdown` | Code-span labels and symbolic/integer count formatting |
| `format_counts_graph_markdown` | Standalone direct-call view from `format_call_graph_markdown` when leaf totals are omitted |
| `format_counts_sigma` | `format_counts_markdown` for standalone leaf totals |

The original Markdown graph assertion now runs through the standalone
production path. Both the direct graph and leaf totals assertions use the
original test functions and fixture sizes. The combined native view still
accepts leaf totals. IDs, default HTML table markup, code-span escaping, and SVG
display wrappers follow native conventions.

## Width and graph admission

The typed factory asks Compiler for its automatic native width model or accepts
an explicit provider. Both paths retain sparse metric visitation. Depth limits
visibility while both queries cover the underlying graph. A provider supplies
the cost for each full-graph node, including hidden nodes. Native program and
connection schedules already have an independent width owner under Q-0806.
They can feed a width table keyed by the rendered resource identities.

The rendering file's explicit graph and cost paths cover arbitrary hashable
graph identities. Automatic inference requires native resource bindings.
Automatic conversion of foreign graphs and arbitrary dtype-specific cache
keys, dirty or shaped allocation identities, and generic control layouts
remain admission boundaries. A missing native width binding raises
`NotImplementedError`. The explicit width or annotation path lets callers
provide measured or modeled widths with their own provenance. Complete node
property callbacks cover the source subclass override use case, with property
values represented as DOT strings or Graphviz HTML.

See `CALL_GRAPH_SOURCE_TESTS.md`, `CALL_GRAPH_NODE_PROPERTIES.md`,
`CALL_GRAPH_COST_DATA.md` and `RESOURCE_CALL_GRAPH_FACTORY.md` for the original
assertions, parsed SVG comparisons, hidden-node failures, cost-cache behavior
and concrete/symbolic resource cases. `CALL_GRAPH_NOTEBOOK.md` records the
complete source/native notebook and nine-node symbolic graph under Q-0738.
Drawing adds zero primitive gates, work wires or measurement events. Emitted
counts and source gate models remain separate in the component evidence.

The concentrated suite passes 4,593 cases across the unchanged-source and
native assertions, Markdown, parsed DOT/SVG, annotation and factory behavior,
the width estimators and program-backed component graphs. It reports eight
source dependency deprecation warnings.
