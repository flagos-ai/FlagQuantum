# Register graph notebook

The locked source is Qualtran v0.7.0 `qualtran/drawing/graphviz.ipynb`, commit
`bcb17bb25360ed68ce3bc084645db3e929799389`, Git blob
`977242ab7fed3bbe5ef11a2befd807007ab69a02`. It has 21 cells, comprising
13 code cells and eight Markdown cells, with no saved outputs or attachments.

The native asset is `examples/notebooks/register_graphs.ipynb`. It composes
existing Core signatures, Compiler register programs, Simulation classical
assignments and Drawer renderers. Numerical execution remains in Simulation.
The shared register graph renderer accepts a private table callback. Explicit
signature metadata supplies presentation without creating another graph IR.

## Complete cell disposition

| Source code cell | Native cell responsibility |
| --- | --- |
| 2 | Native drawing, signature and example imports; explicit SVG display |
| 4 | One-bit atomic metadata instance in compact style |
| 5 | Three-bit parallel-operation metadata instance in compact style |
| 6 | Expand real split / three X atoms / join body in compact style |
| 9 | Render the compact graph style |
| 10 | Render the distinct default graph style |
| 12 | Four-control predicate with original input `[1,1,0,1]`, thirteen producer-value edges |
| 14 | Export the default-style composite DOT |
| 16 | Standalone signature display function replaces the foreign placeholder subclass |
| 17 | Through words of exactly 100 and 200 bits |
| 18 | Same words plus a two-element LEFT two-bit array |
| 19 | Four-element LEFT two-bit array and scalar RIGHT eight-bit output sharing a name |
| 20 | Four-element LEFT two-bit array and two-element RIGHT four-bit array sharing a name |

All eight Markdown cells retain their teaching responsibilities in native terms.
Kernel metadata, unique cell IDs and empty saved outputs are checked. The native
notebook executes independently with reference package imports forbidden and
produces ten SVGs. The original thirteen code cells execute unchanged, with
seven show calls intercepted only to retain their SVGs.

## Signature groups and two styles

`register_graph_table.py` ports the source group ordering and directional
row-span rules. Through ports occupy both columns. Directional ports are paired
within each named signature group, with the final common row spanning any
surplus rows. Single-sided groups retain empty opposite cells. Port text comes
from local register metadata, including row-major multidimensional indices.
Lane identity continues to determine connections.

The default style uses Graphviz table defaults and retains all header/port text.
The compact style uses border zero, cell border one and cell spacing zero,
ten-point headers, suppressed Split/Join text, floating width labels and dot
arrows with size 0.25. Explicit example metadata identifies the actual split/join
instances. Every connection and port survives switching styles.

Tests compare normalized HTML trees against the unchanged source, including
every row, label, port, colspan, rowspan and font attribute. Four original large
signature cases are checked in both styles. Eight additional left/right count
combinations cover both surplus directions, equal sides and single-sided
groups. Both complete composite styles compare every table and edge attribute.
Only generated IDs, HTML case/whitespace and DOT quoting are normalized. The
native example uses `Atom` for the upstream testing fixture `TestAtom`, recorded
explicitly in that comparison. Boundary labels retain native lifetime names,
including `out:` and `call4:reg`. Pixel layout is not an acceptance criterion.

## Executable evidence and resource layers

The parallel example has real split/join SSA calls and three X primitive bodies.
Its five logical instances, eight connections and parallel dependency generation
match the source decomposition. Every complex matrix column is compared to the
unchanged source tensor. An unused spectator remains zero. An arbitrary complex
input and four ordinary append trajectories are checked. Drawing leaves the
physical binding exactly equal. Existing measurement consumers exercise erasure
branches and work-zero behavior in the concentrated regression.

| Layer | Result |
| --- | --- |
| Actual parallel physical instructions | X3 on wires 0,1,2; zero work allocation, measurement and feedback |
| Unchanged source synthetic TestAtom model | T100 per atom, T300 for the parallel example |
| Drawing | Zero added physical instructions |
| 100/200-bit signature examples | Metadata-only display |

The source TestAtom tensor is X while its static cost model deliberately reports
100 T gates. Those two accounting layers remain separate.

## Acceptance scope

Q-0741's complete notebook workflow is accepted after the two-style follow-up.
The foreign object protocol, TypedGraphDrawer and PrettyGraphDrawer subclass
extension surface remain open under Q-0742. This notebook uses native signatures
and component graphs. Its accepted notebook responsibilities do not close that
generic implementation parent.

## Local acceptance

The follow-up adds eighteen cases. All thirty focused cases pass, measuring
100% statement/branch coverage across three owners (141 statements,46 branches).
The graph and typed-call regression passes 1,713 tests in82.60 seconds with eight
upstream parser warnings. Another84 CPU/TN tests pass in16.80 seconds with two
retained warnings. Strict typing covers four Python files. Black, Ruff, Compiler
scope, architecture, multi-level IR and protected API checks pass. All six
reference blobs remain unchanged. Ten native SVGs and both composite-style PNGs
were rendered and inspected. Evidence uses `g2-graph-styles-` in the reference
workspace.

A preliminary broader test selection was stopped after its scope exceeded the
changed owners. Its incomplete run is excluded from acceptance totals. The
historical default-suite timeout remains recorded. The reference environment
uses persistent dependency paths after its old temporary paths disappeared.

Implementation commit `075aade6` completes this follow-up. The ledger has24
accepted parents,1256 open parents and392 active parents, with1187 capability
records. The next batch audits the complete source Graphviz test file.
