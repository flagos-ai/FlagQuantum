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

All three PDFs were rendered with Poppler and inspected at full resolution.
The first render revealed an incorrect table row delimiter and a readout label
crossing quantum feedback; both were corrected. Subsequent review moved the head
feedback label onto the outer vertical route, replaced overlapping feedback
segments with a shared bus, removed implicit table padding, and increased panel
separation. Readout now stays inside the quantum card. The forward path is
horizontal; updates stay below or outside the model. Frozen layers are marked
fixed; only eligible stages receive feedback arrows.

The final PDFs each contain one page. Text bounds are inside the MediaBox;
headings, shapes, route labels and continuation captions are visible without
clipping or overlap. Body annotations use 8pt type (7.97 PDF points); gate
subscripts are naturally smaller. Diagram widths are approximately 18cm; the
four-panel example is approximately 27cm tall. Circuit text is never globally
shrunk. Panels explicitly continue registers without resetting them. The output
is standalone vector PDF, with raster images used only for Notebook/visual QA.

Very long labels and very large register counts can still enlarge a figure;
use overview and a separate circuit source in those cases. Shape declarations
are checked for continuity, not operation formulas, actual gradients or arbitrary
branching graphs. The Notebook independently validates its specific model.

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
