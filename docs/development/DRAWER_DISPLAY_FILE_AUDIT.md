# Notebook display source audit

This batch follows Qualtran v0.7.0 at
`bcb17bb25360ed68ce3bc084645db3e929799389`. The frozen initializer and display
helper blobs are `78692d24b174743b04a4e8c33180d628d14c3be3` and
`93b1a6b6cd1c6c2341789414156479d6a8db9ad4`.

| Source display path | Native owner |
| --- | --- |
| `show_bloq` | `show_register_program` renders a verified ProgramRecord as a width graph, dtype graph or Matplotlib score. Its LaTeX mode accepts the explicit LatexFigure produced by the native circuit drawer. |
| `show_bloqs` | `show_register_programs` displays each graph in an ordered widget with an optional Markdown caption. |
| `show_call_graph` | `show_call_graph` accepts a prebuilt native graph or queries a declared resource call. A depth limit affects visible nodes and preserves complete cost totals. |
| `show_counts_sigma` | `show_counts` renders the shared symbolic totals formatter as Markdown. |
| `show_flame_graph` | `show_flame_graph` uses the existing folded-data producer and an explicitly configured FlameGraph executable. |
| `show_bloq_via_qpic` | `show_latex_diagram` displays PNG bytes from the native TeX renderer and passes the requested notebook dimensions. The figure owns its render cache and temporary files. |

The initializer's three graph classes map to the register and component DOT
exporters. Both endpoint declarations inform dtype labels. Source connection
mismatches remain subject to the owned graph verifier. The classical graph
class maps to the existing classical component graph exporter. The
call-graph class, edge Markdown and leaf sum formatter live in
`call_graph.py`.

The score initializer's RegPosition and LineManager map to
`register_lanes.py` and `RegisterScore`. HLine, VLine and MusicalScoreData
map to the score-data records. WireSymbol, TextBox, Text, RarrowTextBox,
LarrowTextBox, directional_text_box, Circle and ModPlus map to the existing
typed glyph records and register symbol callbacks. get_musical_score_data,
draw_musical_score and dump_musical_score map to the score-data builder,
Matplotlib renderer and JSON exporter. Notebook display lives in
`flagquantum.drawer.notebook_display`, with IPython loaded on demand, and
grouped display additionally needs ipywidgets. Use explicit submodule imports
for these display helpers.

## Verification

The source and native graph displays show the complete eight edges of the
same split/three-atom/join fixture. The native test also retains different
opaque and unsigned word labels across an explicit cast roundtrip.
The parallel fixture's established append regression verifies the full
complex matrix and work restoration. Display invokes no gate emission.
The original fixture's synthetic opaque cost of 300 T and its native three
physical X instructions are recorded separately in the graph notebook audit.

Grouped display preserves caption and diagram order. All four original
resource aggregation modes match the rendered Toffoli resource text.
The prebuilt symbolic graph retains its leaf at a supplied depth of zero,
and symbolic totals match the frozen Markdown output. The pinned FlameGraph
renderer runs on both source and native paths. The native LaTeX path runs
Tectonic and Poppler to produce a real PNG with both default and overridden
display dimensions. An independent Jupyter kernel captures the graph, score,
widget, resource graph, Markdown, flame graph, and PNG MIME outputs.

The source PNG pipeline itself requires qpic, pdfLaTeX and ImageMagick. The
local source probe reaches qpic and records its absence. The source qpic
worker and its wire protocols remain under Q-0750. The native PNG display
contract uses the repository's own TeX figure and has no qpic dependency.
Broader arbitrary source object or NetworkX adapters are outside the native
entry points. The source graph, score, flame and cost internals retain their
individual checklist scopes.
