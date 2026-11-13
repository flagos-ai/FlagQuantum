# Call-graph node presentation callbacks

## Ownership and workflow

Drawer owns call-graph titles, details, node properties and DOT serialization.
Compiler supplies graph identities, multiplicities and costs. The existing
`export_call_graph_dot` accepts optional `details` and `node_properties`
callbacks. Both the concrete modeled factory and typed resource factory forward
them. The title callback remains available on the generic and typed entries.

```python
from flagquantum.drawer.call_graph import export_call_graph_dot, render_call_graph_svg

dot = export_call_graph_dot(
    "root", {("root", "leaf"): 3},
    node_properties=lambda node: {
        "label": node,
        "shape": "box",
        "color": "blue",
    },
)
svg = render_call_graph_svg(dot)
assert svg.startswith(b"<?xml")
assert '"color"="blue"' in dot
```

An explicit details callback selects the rectangular title/detail label from
the locked source base drawer. Titles and details are HTML-escaped. Empty detail
text omits the line and font elements. Without details or complete properties,
the existing annotation-table presentation applies.

A complete property callback replaces all default node properties. It bypasses
title, detail and annotation formatting, matching a source subclass override.
Callbacks run once per visible node in topological order. Factory depth cutoffs
retain full cost and width evaluation while hidden nodes receive no display
callbacks. Callback failures propagate and a retry owns fresh local state.
Input property mappings and annotation tables remain caller-owned.

Property keys and ordinary values are quoted with the existing DOT string
escaping rules. A value delimited by angle brackets is trusted Graphviz HTML
markup, including labels and external labels. Property values use strings.
This is the native representation of the source property dictionary boundary.
Default source font, padding and table presentation retain their documented
native adaptation. Callback dictionaries do not alter graph identities or edges.

## Locked source comparison

Qualtran v0.7.0 commit `bcb17bb25360ed68ce3bc084645db3e929799389` remains
unchanged. The `drawing/bloq_counts_graph.py` Git blob is checked against the
fixed inventory before tests. Apache-2.0 attribution remains in the native owner.

| Source method | Native execution boundary |
| --- | --- |
| `_CallGraphDrawerBase.get_node_title` | Existing title callback |
| `_CallGraphDrawerBase.get_node_details` | Optional details callback |
| `_CallGraphDrawerBase.get_node_properties` | Exact title/detail label construction |
| `GraphvizCallGraph.get_node_properties` subclass override | Complete property callback replacing the default node attributes |
| Base `add_nodes` and `get_graph` | Existing topological traversal and DOT serializer |
| Base `get_svg_bytes` | Existing bounded Graphviz renderer |

Twelve cases compare escaped and multiline titles/details with actual unchanged
source base methods. Six property sets compare complete dictionaries with real
source subclass overrides, including shape, colors, fill, empty labels, quoted
text, newline/backslash tooltips, HTML labels and external labels. Both paths
render actual SVG and compare all visible text. Additional cases cover value
escaping against extra-node/edge creation, all factory depths, callback order,
complete override precedence, identity-preserving failure propagation, retry
and cycle rejection before presentation.

## Acceptance and resource accounting

29 focused cases pass in 5.22 seconds. Concentrated regression passes 1419
cases in 37.81 seconds. Independent real-kernel notebook/CPU/TN integration
passes 87 cases in 13.25 seconds. There are zero skips. Focused/concentrated
runs expose eight/1112 existing source/dependency warnings and integration
retains two existing warnings. Native Kickmix Simulator and the locked
Qualtran/quimb versions are verified before regression.

Strict typing covers both changed Python files. Black, Ruff, Compiler scope,
architecture, internal IR and protected API checks pass. Nine module doctest
examples and the Markdown workflow pass. Coverage is 100 percent for the existing
example, 96 percent for Drawer call_graph and 96 percent combined. The property
serializer and all three graph-export entries have complete measured statement
and branch coverage. Remaining misses belong to existing Markdown edge-label
and renderer failure paths.

Initial test setup omitted source edge multiplicities, and direct subclassing
of untyped external classes failed strict checking. The reference boundary now
creates real source subclasses through `type`, and source graphs have their
original edge weights. One external-class variable naming diagnostic and the
quoted-key doctest expectation were also corrected. The source implementation
and original tests were unchanged. Logs retain the initial diagnostics.

| Accounting category | Observed or declared value |
| --- | --- |
| Presentation added primitive gates / work wires / measurements | 0 / 0 / 0 |
| Approximate three-bit swap declared categories | T12, Clifford65 |
| Exact Fredkin emitted primitives | CX8, H2, T4, Tdg3, work0, measurement0 |
| Exact Fredkin terminal-T / default QEC estimate | 7T / 4T |
| Existing measured merged-Fredkin emitted T-equivalent / measurement | 11 / 1 |
| Merged model components | 4 + 7 + 0 |

Existing exact sequence, complex-amplitude, measurement-branch, work-zero and
`circuit.append(op)` regressions pass in the concentrated suite. Presentation
consumes declarations without executing a quantum operation.

Q-0734.05 records the property capability. Q-0734 remains open for universal
width admission. Foreign graph admission, CDDL renderer packaging, browser
interaction, historical QFT 190 versus 496 and the historical full-default
600-second timeout retain their independent open records.
