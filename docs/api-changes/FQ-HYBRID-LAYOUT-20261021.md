# Explicit hybrid layers, readout semantics and readable circuit panels

## Authorization and scope

On 2026-10-21 the user authorized these three draw_hybrid improvements, an
updated executable CNN/quantum Notebook, compiled and visually reviewed PDFs,
tests and a new PR. This decision authorizes the additive drawer namespace type
and keyword arguments below and removal of invented default feature counts.
Owner: compiler/drawer. No Core IR, model execution or root exports change.

## Interface and compatibility

`ClassicalLayer(label, input_shape, output_shape, trainable=False)` is immutable
presentation data. Shapes contain positive, batch-free integer dimensions.
Existing IR and nn.Module types do not describe classical diagram labels and
shapes; this record describes presentation only and cannot execute a model.

`draw_hybrid` keeps accepting string encoder/head labels, or accepts nonempty
sequences of ClassicalLayer. It validates adjacent shapes, the encoder's final
vector against encoder_features, and readout_features against the head input.
Detailed heads require an explicit readout label and feature count. Shape checks
verify the supplied description; they do not infer layer mathematics or trace
arbitrary PyTorch code. The Notebook separately checks actual layer outputs.

The encoder_features/head_outputs defaults become None: detailed layers derive
them; string stages display unspecified counts. Existing explicitly supplied
counts and labels retain their meaning. Previously omitted counts displayed
fictional example values (32 and 10); callers wanting those must supply them.

New keywords: readout, readout_features, encoder_trainable, quantum_trainable,
head_trainable, quantum_view='auto'|'overview'|'detail', gates_per_panel=8.
Unspecified readout remains visibly unspecified with a data arrow; an explicit
readout uses a double line. Declared readout is not an inferred measurement.
Frozen stages have no optimizer update arrow; stage flags override layer flags.
The drawing never runs a forward pass, computes gradients or mutates parameters.

Short circuits can appear inline at natural font size. Auto selects separate
ordered panels for larger circuits; overview omits circuit detail, and detail
always shows panels. Panel boundaries continue the same registers, not reset
states. Final observable/readout annotations appear only in the last panel.
No entire circuit is shrunk to fit an architecture inset. No new dependency,
resource license change, protected snapshot regeneration or executable IR exists.

## Acceptance

Preserve old explicit-call behavior, reject shape and semantic conflicts, check
frozen-stage feedback, test immutable input ownership and all layout modes,
compile real PDFs for detailed CNN and long-circuit layouts, execute the Notebook
including actual shapes and gradients, render every final PDF page and inspect
labels, routing, font sizes, margins and continuation semantics. Test source-only
work without external tools and run repository quality/default/runtime gates.

## Publication layout acceptance amendment

The user clarified on 2026-10-21 that generated figures must be directly usable
in papers; demonstration-quality output is insufficient. The existing PR's
presentation is revised before merge, with no new public signature or root API.
Owner: compiler/drawer; authorization: the user's explicit publication standard.

The default hybrid figure targets a 170 mm two-column paper text area. Its
content canvas is 166 mm wide with 2 pt export margins and 9 pt body/gate text.
No global scaling or circuit inset is used. Auto and detail both separate the
architecture (panel a) from the exact circuit (panel b and continuations), even
for short circuits. Overview retains only panel a. An empty circuit is explicit.
Demo titles, redundant state labels and color legends move out of the artwork;
notation, omitted batch axes and continuation semantics belong in the caption.
Trainability and unspecified data remain visible. Long circuit panels fail
compilation with an actionable width diagnostic rather than silently shrinking
text or producing an oversized figure; reduce gates_per_panel or use overview.

Existing arguments retain their semantic meanings; auto's placement is a
presentation change in this unmerged feature. Standard LaTeX drawing also gets
2 pt crop margins. Default terminal text and Matplotlib behavior are unchanged.
The Notebook must demonstrate final-size paper inclusion with a suitable caption.
Acceptance requires rendered size/font and grayscale checks, not only successful
compilation. This targets two-column-width figures; shrinking that asset to a
single column is not a supported readability claim.


## Journal-style composition revision

The user rejected the framed flowchart aesthetic and explicitly requested a
Nature/Science-style figure on 2026-10-21. This supersedes the preceding visual
acceptance amendment; it does not change public signatures, validation or model
semantics. Owner: compiler/drawer. The prior 170 mm/9 pt nested-card rendering
met mechanical checks but failed the user's visual quality requirement.

Use Nature's official research figure guide as a measurable reference, not as a
claim of journal endorsement: 183 mm double-column width, compact arrangements,
standard sans-serif type, 5-7 pt labels, editable vector output, black text and
accessible colors. Source:
https://research-figure-guide.nature.com/figures/building-and-exporting-figure-panels/
Science's current author page could not be retrieved (HTTP 403); no unverified
Science-specific measurements are asserted.

The default becomes a horizontal sequence of schematic tensor glyphs, actual
quantum gates when short enough, declared readout and classical output. Glyphs
are schematic; numeric shape labels remain authoritative. Avoid nested cards,
redundant stage titles and large empty continuation panels. Auto inlines up to
eight instructions on at most four wires at natural 7 pt size. Detail forces
continuation panels; compact panels share rows when their measured widths fit.
Overview retains a unitary summary. All modes retain exact IR validation and
terminal readout annotations; panels continue the same registers. Rendering
still fails rather than shrinking over-wide content. Training remains an
explicit opt-in; main-paper Notebook artwork presents the forward architecture,
while the frozen example demonstrates the optional update paths.

The content canvas is 181.5 mm plus 2 pt export margins (about 183 mm overall).
The earlier 166 mm guard and associated layout dimensions are superseded.
Long classical chains may need explicit summary labels rather than every
intermediate tensor; the drawer must never invent encoder mathematics.
The default install still exports source without a compiler; rendering uses
standard local TeX fonts. No new Python dependencies or licensed source edits.
