# Native register graph notebook

The locked Qualtran v0.7.0 `drawing/graphviz.ipynb` contains 21 cells: thirteen
code cells and eight Markdown cells. Its blob is
`977242ab7fed3bbe5ef11a2befd807007ab69a02` at source commit
`bcb17bb25360ed68ce3bc084645db3e929799389`. All original code strings execute
without edits, alongside the standalone native
[`register_graphs.ipynb`](../../examples/notebooks/register_graphs.ipynb).
The original file has no saved outputs, attachments or additional assets.

## Responsibilities

`drawer/signature_graph.py` projects standalone signature metadata into the
existing RegisterScore and DOT renderer. It does not fabricate executable
function bodies for placeholders. Opposite-side same-name ports retain different
lifetimes; an output-only name receives `out:` prefixes if needed. RIGHT-before-
LEFT ordering is tested independently. THRU words retain one lane. Arrays retain
row-major element order and original widths. Signature drawing admits positive
concrete quantum words only; symbolic/classical and nonpositive widths remain
refused under the established drawing profile.

`examples/algorithms/register_graphs.py` composes real Compiler split/join/X
programs and the existing predicate ladder. Simulation computes classical
assignments; Drawer receives them explicitly. A small example style callback
names reshaping operations Split/Join without altering their connectivity.
There is no foreign object dispatcher, numerical implementation in Drawer,
second graph representation or protected API change.

## Complete cell disposition

| Source code cell | Native cell responsibility |
| --- | --- |
| 2 | Native drawing, signature and example imports; explicit SVG display |
| 4 | One-bit atomic metadata instance |
| 5 | Three-bit parallel-operation metadata instance |
| 6 | Expand real split / three X atoms / join body |
| 9 | Render the shared native table view |
| 10 | Inspect native DOT, adapting the source default-versus-pretty style demonstration |
| 12 | Four-control predicate with original input `[1,1,0,1]`, thirteen producer-value edges |
| 14 | Export the actual composite DOT |
| 16 | Standalone signature display function, replacing a foreign placeholder subclass |
| 17 | Through words of exactly 100 and 200 bits |
| 18 | Same words plus a two-element LEFT two-bit array |
| 19 | Four-element LEFT two-bit array and scalar RIGHT eight-bit output with the same name |
| 20 | Four-element LEFT two-bit array and two-element RIGHT four-bit array with the same name |

All eight Markdown cells retain their teaching responsibilities in native terms.
The native kernel metadata, unique cell IDs and empty saved outputs are checked.
The native notebook is independently executed with reference package imports
forbidden. Nine native SVGs are produced; all original show calls render real
source SVGs, and the original default/pretty SVG-returning cells execute too.

The source's two complete GraphDrawer style classes are not introduced here.
Native cell10 uses inspectable DOT instead of reproducing that second visual
style. Q-0741 therefore remains open for that explicit notebook demonstration;
all cells have an execution/disposition, but the original two-style capability
is not marked ported. Foreign class discovery and full PrettyGraphDrawer
behavior remain open under their implementation parents as well. Port labels
use native lifetime identities rather than claiming source display spelling or
pixel equality. The four signature cases compare every connection width and
boundary port count/direction against the actual unchanged source graph.

## Executable evidence and resource layers

The parallel example has real split/join SSA calls and three X primitive bodies.
Its five logical instances, eight connections and parallel dependency generation
match the source decomposition. Every complex matrix column is compared to the
unchanged source tensor; an unused spectator remains zero, and an arbitrary
complex input and four ordinary append trajectories are checked. There are no
measurements in this example. Existing predicate/measurement consumers in the
regression separately cover measured erasure branches and work zero.

| Layer | Result |
| --- | --- |
| Actual parallel physical instructions | X3 on wires0,1,2; no work allocation, measurement or feedback |
| Unchanged source synthetic TestAtom model | T100 per atom, T300 for the parallel example |
| Drawing | Adds no instructions; binding before/after drawing is exactly equal |
| 100/200-bit signature examples | Metadata only; no unitary, execution or resource model claimed |

The source TestAtom tensor is X while its static cost model deliberately reports
100 T gates. Neither source is modified to make these different accounting
layers equal. Model T300 is not a claim that native X3 has that physical cost.

## Remaining scope

Q-0741's default/pretty style distinction remains explicit. No generic upstream
Bloq or pydot production wrapper is introduced. Existing renderers, Core
signature namespaces and Compiler execution boundaries retain authority. The
historical default-suite timeout is not repeated by this notebook batch.

## Local acceptance

Implementation `1fbfdecb` adds twelve cases. The concentrated regression passes
2,931 tests in68.37seconds (35 retained warnings), plus84 CPU/TN tests in
5.86seconds (two warnings). Final focused coverage passes12 cases in5.55seconds;
both new Python owners have100% statement/branch coverage (83 statements,
22 branches). Strict typing covers three Python files; Black, Ruff, Compiler
scope, architecture, multi-level IR and protected API checks pass. Six source
blobs match the frozen inventory. All13 native cells compile and independently
execute; the nine SVGs are saved and the composite/last large-signature PNGs
were inspected. The existing measurement-uncomputation example also runs.
Evidence uses `g2-graphviz-notebook-` in the reference workspace. Default-suite
historical failures/timeouts and the still-open style distinction are retained.
