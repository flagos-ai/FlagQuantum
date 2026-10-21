# Hybrid layer and circuit-layout validation

## Scope

The user authorized explicit classical layers/shapes, readout and trainability
semantics, and readable overview/detail layout on 2026-10-21. This builds on
PR #668. It changes presentation only: no model tracing, operation execution,
Core IR fields, numerical kernels or root exports change. The
[API decision](../api-changes/FQ-HYBRID-LAYOUT-20261021.md) records the authorized
interface and the migration from fictional count defaults to unspecified counts.

## Executed Notebook

`examples/notebooks/hybrid_latex_drawing.ipynb` executes without errors and retains
five image outputs: existing Matplotlib, native LaTeX circuit, detailed CNN hybrid,
frozen-encoder overview and long-circuit continuation panels.

Actual encoder outputs, excluding batch:

| Operation | Input | Output | Trainable parameters |
| --- | --- | --- | --- |
| Conv2d 3x3 | 1 x 8 x 8 | 3 x 8 x 8 | Yes |
| Tanh | 3 x 8 x 8 | 3 x 8 x 8 | No |
| AdaptiveAvgPool2d | 3 x 8 x 8 | 3 x 1 x 1 | No |
| Flatten | 3 x 1 x 1 | 3 | No |
| Z readout | Three-qubit state | 3 | Circuit parameters |
| Linear | 3 | 2 | Yes |

Every actual layer output is compared with its ClassicalLayer declaration.
The three Z values come from the same circuit sequence shown in the detailed
figure. The synthetic joint training step records gradient norms 0.003033
(encoder), 0.001278 (quantum) and 0.209511 (head), and loss 0.707551. An Adam step
changes quantum parameters. A separate frozen copy produces no encoder parameter
gradients, retains finite nonzero quantum/head gradients, and performs a step
with only trainable parameters. The matching overview omits encoder feedback.
These checks establish this small CPU example, not model accuracy or scaling.

## PDF and visual review

The Notebook itself exports three PDFs and matching editable TeX/PNG files:

- `cnn-hybrid-detail.pdf`: actual four-layer encoder, declared Z readout, head,
  training feedback and the eight-gate circuit detail.
- `cnn-hybrid-overview.pdf`: the frozen encoder with quantum/head feedback and
  a compact qubit/gate-count summary.
- `long-circuit-hybrid.pdf`: separate layout stress example, 24 rotations in four
  ordered six-gate panels. This is explicitly not the trained model's circuit.

The user subsequently required publication-ready output as the default acceptance
standard. The initial 18 cm demo layouts with 8 pt labels were rejected and
replaced: the current hybrid PDFs use a 166 mm content canvas with 2 pt crop
margins (about 167.4 mm overall), and 9 pt body/gate text. Architecture (a) and
exact circuit (b) remain separate even for short circuits. Explicit paragraph
spacing separates the panels; TeX table row spacing had been insufficient for
tall nested diagrams. The head feedback route now derives its outside port from
the head boundary rather than the narrower optimizer boundary.

Demo titles, duplicate trainable/fixed lines and in-graphic color legends were
removed. Trainable layers/circuit parameters carry a star; dashed routes remain
explicitly labeled. Input/output shapes and declared/unspecified readout remain
visible. Captions in the Notebook define x/z, stars, batch-free shapes and
continuation semantics. A ready-to-use figure* snippet includes the PDF in a
170 mm text area; the intended use is a two-column-width figure, not shrinking
that asset into an 85 mm column. The four-panel stress example is supplementary
material, not a replacement for a compact main-paper overview.

Every panel is measured by TeX at the intended size. Content wider than 166 mm
raises a rendering error with instructions to reduce gates_per_panel, shorten
labels or choose overview. The drawer never globally scales down a circuit.
Large register counts can still require a tall supplementary figure. Native
LaTeX circuit exports also use 2 pt crop margins. All source remains editable.

Final-size/color/grayscale inspection and exact dimensions are recorded below.
No training computation, dependency, licensed runtime or protected API changes.

## Tests and packaging

Focused tests cover valid CNN workflows, shape/readout conflicts, unknown counts,
freezing, immutable input ownership, label escaping, invalid options, supported
and rejected circuit semantics, short/empty/long circuits, all layout modes,
ordered continuation, and terminal requests emitted once. Existing drawer,
Notebook formatter, resource/license, missing-tool/cache and installation/lane
policy tests are included. A real-rendering regression compares PNG dimensions:
circuit panels must increase height, not create side-by-side columns.

Measured focused and broader-tier results are recorded in the PR description.
The changed hybrid modules are checked with strict mypy, Ruff and Black. Source
hygiene, architecture, links, language and the protected API baseline are checked
without regenerating snapshots. No new dependency or license resource is added.

A local macOS source checkout initially lacked its native extension; after
building it in place, its OpenMP library also needed the virtual environment's
Torch lib directory on DYLD_LIBRARY_PATH. The independent QFT fusion test then
passes in both precisions. This is a local loader setup, not a simulation patch.
The complete tier reruns use that environment and one thread per worker.

## Initial implementation results (before publication-layout revision)

- Focused drawer/legacy/docstring/installation/lane-policy selection: **104
  passed**, including three real TeX/PNG integration tests.
- Hybrid modules: **100% statement and branch coverage**, 130 statements and
  72 branches; the coverage selection has 64 passing tests. This is not a
  whole-package coverage claim.
- Whole-package strict mypy: **776 source files pass**. Ruff, Black, source
  hygiene, architecture, language, local links and protected API baseline pass.
- Complete `pr-runtime`: **1718 passed, 123 skipped**, 168.82 seconds.
- Complete `pr-default` on feature code `430901a3e`: **14128 passed, 165
  skipped**, 2979 deselected, 791.17 seconds. It runs every selected test,
  without per-test exclusions, in a single-branch validation clone with the
  package installed and its native extension available.
- Main advanced to `362b2184a` during validation. Its four changed files contain
  tests, fixture data and documentation only. After merging that baseline, all
  **44 newly introduced matrix expectation contract tests pass**. No production
  source from that synchronization changes the tested drawing implementation.
- Wheel/sdist build; the installed wheel's hybrid Python files match the
  checkout. Outside the repository it imports ClassicalLayer, validates shapes,
  exports the licensed project and compiles a real PDF.

The single-branch clone avoids a host-specific Git topology: this machine has an
unrelated local branch containing the intentionally absent producer revision
used by the provenance negative test. No history, artifact or test assertion is
changed to suppress that test. A 0.2-second calibration-budget test also failed
during the earlier parallel run; its isolated rerun and final complete serial
tier pass. The clean validation clone initially lacked installed entry-point
metadata; installing the built wheel resolved that setup error, and its discovery
test passes in the complete tier. These are recorded as environment constraints,
not drawing fixes or weakened checks.


## Earlier publication-layout verification (superseded composition)

The final default exports measure 167.41 mm wide including crop margins:
CNN detail 104.30 mm high, frozen overview 75.80 mm high, and the four-panel
supplementary example 189.80 mm high. Body/gate text measures 8.97 PDF points
(9 TeX points); mathematical subscripts retain their normal smaller size.
Rotated feedback annotations use the same 9 pt font. Every text bounding box
lies within the page. Each PDF is one vector page. All three latest PNGs and a
grayscale CNN rendering are inspected for clipping, crossing routes, spacing
and contrast. The default CNN PDF is also embedded, without artwork editing,
into an actual 170 mm-wide figure* in a two-column article, with a separate
caption; that compiled article contains vector artwork, not a raster image.

The Notebook executes without errors and retains five image outputs, actual
joint/frozen training checks, and a ready-to-use paper-inclusion TeX snippet.
The revised artwork no longer requires removing a demo title/legend, shrinking
a circuit or rerouting feedback after export. Main-paper and supplementary
roles are explained in the Notebook; the synthetic model is not presented as
an experimentally validated classifier.

- Focused selection: **105 passed**, including four real compiler/converter
  tests and an intentionally over-wide parameter label that must fail rather
  than shrink. Existing text/Matplotlib/source/cache behavior remains covered.
- The two hybrid modules: **100% statement and branch coverage**, 135 statements
  and 76 branches; 63 tests in the coverage selection.
- Whole-package strict mypy: **776 source files pass**. The complete pre-commit
  suite passes, including source hygiene, English content, architecture, docs,
  protected API baseline, Ruff, Black and package/tooling type checks.
- Wheel and sdist build. Outside the checkout, the installed wheel's four drawer
  Python files match the source, compiles PDF/PNG within the paper width and
  exports the complete licensed TeX bundle.

The full default tier uses the same isolated validation checkout as above. A
local fetch could not transport a missing promisor object, so the exact source
snapshot was copied and committed there; equality of the Git tree object was
checked before testing (3fa22e9056094962421ec9ef743975d3ff79a988).
No test exclusions, source-history rewrites or provenance-fixture changes are
used. The original shared checkout and other user branches remain untouched.


The current pr-default selection is fully verified in two contiguous segments:
**14,172 passed, 165 skipped** in total (2,980 deselected by the tier's original
marker). The initial sandboxed run has 10,003 passed and 95 skipped before a
Gloo setup error, `Cannot resolve 127.0.0.1 to a (local) address`. Without changing
source or tests, the six-case failing module passes with local-network access.
A fresh collection identifies the exact remaining 4,239 node IDs beginning with
the interrupted case; the already-completed prefix is verified against the
initial counts, accounting for 48 collection-time skips. That complete suffix
then passes: 4,169 passed, 70 skipped, 210.06 seconds. No cases are omitted or
assertions weakened. This is full selection coverage across an environment-error
recovery, not a claim of one uninterrupted full-tier invocation.

The complete current pr-runtime invocation passes: **1,720 passed, 123 skipped**,
15,474 deselected, 198.58 seconds. It runs with local-network access for its
original integration cases and uses unchanged assertions and marker selection.
Only this evidence record and the manual known-limitations paragraph change
after the verified implementation snapshot.


## Journal-style composition revision

The user rejected the nested-card appearance, including its whitespace and
isolated circuit detail, despite the earlier mechanical checks. The default is
now a horizontal sequence of schematic feature maps/vectors, the exact short
circuit, declared readout and output. The main Notebook figure uses training=False
for forward-model artwork; actual training and gradient checks remain intact.
The frozen example separately exercises training=True and its update ownership.
Stars mark declared trainable components. Glyph geometry is schematic; numeric
shapes, not plane size or the number of representative tiles, carry dimensions.

The measurable design reference is Nature's official research figure guide:
https://research-figure-guide.nature.com/figures/building-and-exporting-figure-panels/
This is not a claim of endorsement or submission acceptance. The Science author
page returned HTTP 403, so no unverified Science-specific dimensions are used.
The default content canvas is 181.5 mm plus 2 pt margins (182.9 mm total), with
7 pt Helvetica-compatible Nimbus Sans text and conventional mathematical scripts.
Panels are measured before arranging them in two columns; wider panels stack
without scaling. A complete figure exceeding 170 mm height fails with guidance
to use an overview or separate detail. The earlier 166 mm/9 pt output is obsolete.

Visual iteration fixed a silent XeTeX TU-encoding fallback to Roman: explicit T1
font selection now embeds Nimbus Sans regular and bold. Named TeX box registers
prevent nested minipages and TikZ from clobbering raw scratch registers. Paragraph
starts center circuit boxes within their columns. Tensor glyph ports and the
actual circuit center share the forward baseline; metadata below the circuit
cannot move that baseline. A trailing space in the outer figure box was caught
by the real-rendering width limit and removed. The declared classical readout
uses a double-line connection after the readout stage.

Model validation, IR ownership, unsupported-operation handling and public
signatures are unchanged. No Python dependency or upstream resource/license
changes; the owned style adds standard TeX font loading and private box storage.
The layout implementation has one private module, keeping validation separate
from the tensor/circuit composition. The Notebook labels operations concisely
(Conv 3x3 and Avg. pool), with actual PyTorch shapes still checked independently.
