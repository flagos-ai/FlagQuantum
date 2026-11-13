# Complete call-graph source test audit

The locked `qualtran/drawing/bloq_counts_graph_test.py` contains seven tests and
one node-label collection helper. `tests/unit/test_call_graph_source_tests.py`
executes every test twice, once unchanged against the source and once with
native producers substituted at the test boundary. Both executions retain the
original function code objects. The helper's original random label shuffle also
runs through the native inspection view.

The reference version is Qualtran v0.7.0 at
`bcb17bb25360ed68ce3bc084645db3e929799389`. Six source blobs are verified against
the immutable inventory before any source test executes. The native production
workflow remains `python -m examples.algorithms.resource_call_graphs`.

## Per-definition disposition

| Source definition | Native execution |
| --- | --- |
| test_format_counts_sigma | Shared conjunction recipe and graph leaf counts feed the existing combined Markdown formatter's totals section |
| test_format_counts_graph_markdown | The same native graph feeds its direct-call section |
| _get_node_labels_from_pydot_graph | Original helper consumes a test-only native DOT inspection view, retaining random.shuffle |
| test_graphviz_call_graph_no_data | Existing native symbolic graph and DOT exporter compute all node labels |
| test_graphviz_call_graph_with_data | Explicit caller annotations pass through the existing native DOT exporter |
| test_graphviz_call_graph_from_bloq | Native declaration fixture, QEC query cache and declared sequential width callback compose annotated DOT |
| test_graphviz_call_graph_from_bloq_agg | Existing t_and_ccz_budget supplies both aggregate columns |
| test_graphviz_call_graph_from_bloq_max_depth | Native graph visibility gives the original three and two generation assertions |

## Ownership and boundaries

Compiler owns conjunction expansion, declared graph traversal, symbolic leaf
multiplicities, generic cost caching and QEC categories. Drawer owns Markdown,
DOT, title callbacks and annotation mapping. This is single_device_fast_path
presentation verification with no register outputs, bit order or work-wire
ownership. It creates no production dependency or resource identity.

The test-only inspection view provides the source consumer's method shape and
converts native tables to the source's outer font, padding, tag case and newline
presentation. It parses and preserves every computed native title and cell
value. Native DOT remains unchanged. Reference-specific names occur only at the
test title boundary. There is no production foreign graph adapter.

The test-only factory composes native owners for the exact source fixture.
Its explicit width callback takes the maximum of declared node and callee widths
through the existing generic cost query. This fixture has zero bystanders and
sequential declared width requirements. The callback does not admit arbitrary
foreign graph liveness. Its declarations are independently constructed by the
native example. Explicit user-supplied annotation tests consume caller values,
while factory cost tests compute their own native costs. Generalized production
factory composition remains owned by Q-0734.

Eight additional cases compare every visible table at depths zero, one, two and
unlimited with both factored and T/CCZ annotations. They verify node counts,
edge counts, topological generations and exact cell text after the stated HTML
presentation adaptation. Original root spot-check assertions are supplemented
by whole-graph comparisons. An incorrect native root cost of 100*n+601 must fail
the original unchanged assertion. This negative control verifies that the
presentation adapter cannot replace calculated costs with expected values.

## Resource accounting

The demonstration's declared atomic model is 100T and the root model is
100*n+600T. Its source numerical atomic tensor implements X. The declaration
and physical action remain separate evidence. Drawing adds zero primitive gates,
work wires and measurements. Existing exact Fredkin and measured merged-Fredkin
complex-amplitude, branch cleanup and circuit.append regressions remain in the
concentrated suite. Physical counts and model estimates are recorded separately
in RESOURCE_FLAME_GRAPH.md.

The initial command used the workspace parent as its working directory and
created no files. It was rerun in the repository after a successful scope check.
Strict typing then required widening the conjunction root to the existing
ResourceCall union before generic graph inference. Both diagnostics are retained
in the batch evidence. Source code and original assertions were unchanged.

## Batch acceptance

Implementation 6c4d2333 passes 23 focused cases in 1.64 seconds, 1248 concentrated
cases in 26.63 seconds and 87 independent notebook/CPU/TN integration cases in
11.40 seconds. There are zero skips. Focused and concentrated runs expose eight
and 1112 dependency/source warnings respectively. Integration retains two
existing warnings. Strict typing of the new test file, Black, Ruff, Compiler
scope, architecture, internal IR and protected API checks pass.

The retained measured production scope covers the existing call-graph example
at 100 percent, Drawer call_graph at 92 percent and both together at 93 percent.
The uncovered Drawer lines are existing Markdown edge-label, invalid aggregation
and renderer error paths. No production code changes were needed in this batch.
The primary ledger closes the complete test asset Q-0735 with Q-0735.03.
Q-0734 retains the generic implementation and factory-admission work. Counts
are 32 accepted parents, 1248 open, 387 active and 1195 capability records,
comprising 1194 current and one historical.
