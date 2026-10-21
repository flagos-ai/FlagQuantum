# Circuit and hybrid-model drawing

The drawer owns presentation, source export and optional image rendering. It
never executes a model, compiles quantum gates, chooses devices or submits jobs.
Circuit semantics come from Core IR and its operator schema. Existing legacy
operation-history adaptation stays in `ir_adapter.py` for text/Matplotlib.

## Existing workflows

```python
import flagquantum as fq

circuit = fq.Circuit(2).h(0).cx(0, 1)
print(circuit.draw())
fig, ax = circuit.draw(format="mpl")  # pip install "flagquantum[viz]"
fig.savefig("circuit.pdf", bbox_inches="tight")
```

## LaTeX source and Notebook display

```python
figure = circuit.draw(format="latex")
source = figure.source                    # no external tools needed
figure.save_source("circuit.tex")
figure.export_bundle("overleaf-circuit")  # runtime and component licenses included
figure                                   # final Notebook expression displays image
```

`str(figure)` is the source. `standalone=False` returns an embeddable source
fragment; rendering and bundle export wrap it in a standalone document.
`display=True` explicitly invokes IPython display and returns the same object.
In a Python script use `figure.display()` for an IPython display call.

```python
figure.save("circuit.pdf")
figure.save("circuit.svg")
figure.save("circuit.png")
pdf_bytes = figure.render("pdf")
```

| Capability | Dependencies | Support boundary |
| --- | --- | --- |
| Text circuit | Default FlagQuantum install | Existing Circuit/IR/qdev adaptation |
| Matplotlib circuit | `flagquantum[viz]` | Existing figure/axes result |
| LaTeX source / portable project | Default FlagQuantum install | Native Circuit/IR/serialized IR, scalar and symbolic unitary gates |
| PDF compilation | Tectonic or TeX Live on PATH | Temporary isolated build, shell escape disabled |
| SVG display/export | PDF compiler + Poppler `pdftocairo` | Preferred Notebook MIME when available |
| PNG display/export | PDF compiler + Poppler `pdftoppm` or `pdftocairo` | Notebook MIME when SVG converter is absent |
| Explicit hybrid view | Same source/render dependencies | Encoder -> circuit -> head, optional training paths |

Install a compiler and Poppler using the platform's package manager (for example,
`brew install tectonic poppler` on macOS). A preinstalled TeX Live engine
`pdflatex`, `xelatex` or `lualatex` also works. The kernel process must see tools
on PATH; installing them on a different Jupyter frontend machine is insufficient.
`engine="/path/to/tectonic"` selects a compiler explicitly; `timeout=120.0`
bounds each external command. First-use Tectonic resource downloads are owned by
Tectonic. FlagQuantum does not download or install executables.

The `LatexFigure` MIME protocol does not require importing IPython at source
export time. Only explicit `.display()` / `display=True` imports IPython.
The `viz` extra includes IPython for explicit display; an existing Notebook
kernel already provides it. LaTeX source and bundle export need no extra.
Compiled PDF and converted images are cached on that immutable figure instance;
there is no global persistent cache. A new figure captures changed model
parameters. Missing tools and failed compilation raise actionable errors and
never substitute another renderer or circuit. Bundle export refuses to overwrite
existing files with different contents.

## Explicit hybrid architecture

```python
from flagquantum.drawer import ClassicalLayer, draw_hybrid

encoder = (
    ClassicalLayer("Conv2d", (1, 8, 8), (3, 8, 8), trainable=True),
    ClassicalLayer("Pool", (3, 8, 8), (3, 1, 1)),
    ClassicalLayer("Flatten", (3, 1, 1), (3,)),
)
head = (ClassicalLayer("Linear", (3,), (2,), trainable=True),)
hybrid_circuit = fq.Circuit(3).ry(0, fq.Parameter("theta_0"))
figure = draw_hybrid(
    hybrid_circuit,
    encoder=encoder,
    head=head,
    readout="Z expectations",
    readout_features=3,
    training=True,
    quantum_view="detail",
)
figure
```

ClassicalLayer is immutable presentation data, with shapes excluding the batch
axis. Adjacent layer shapes must agree. The encoder's final vector determines
encoder_features, and the head's final vector determines head_outputs; conflicting
explicit counts raise ValueError. A detailed head requires a declared readout and
readout_features matching its input vector. These checks validate declarations,
not operation formulas or actual model outputs. The Notebook checks both.

String encoder/head labels remain supported. Omitted counts now display
unspecified values instead of the former example defaults of 32 and 10. Pass
those counts explicitly when they describe your model. Readout is never guessed
from the number of qubits or encoded features; unspecified readout uses a normal
data arrow. An explicit readout label uses a double line, as a caller declaration.

With training=True, parameter feedback is drawn only for trainable stages.
Detailed classical stages must contain a trainable layer. encoder_trainable=False
and head_trainable=False freeze entire stages and override per-layer labels.
quantum_trainable=False freezes the circuit; circuits without gate parameters
receive no quantum update. Fixed transformations still propagate input gradients:
their label does not imply detachment. Drawing never executes a training step.

The default hybrid renderer composes a horizontal journal-style schematic:
representative tensor planes/vectors, the actual short quantum circuit, declared
readout and classical output. Layer labels and numeric shapes carry the model
meaning; glyph geometry is schematic and never depicts measured activations.
Names and black text preserve meaning in grayscale. There are no nested stage
cards. Trainable components carry stars; the paper caption defines that notation.
Use training=False for forward-model artwork; training=True adds the optional
optimizer and update paths. Unstarred transformations still propagate gradients.

The typography/size reference is Nature's
[official research figure guide](https://research-figure-guide.nature.com/figures/building-and-exporting-figure-panels/):
183 mm double-column width, a maximum 170 mm figure height, compact arrangements
and standard sans-serif labels. The content canvas is 181.5 mm plus 2 pt crop
margins (182.9 mm overall). Text uses 7 pt Helvetica-compatible Nimbus Sans via
standard TeX fonts. Mathematical notation retains its normal smaller scripts.
This is a design reference, not journal endorsement or guaranteed acceptance.

quantum_view="auto" integrates at most eight instructions on at most four qubits
at natural size. Larger circuits use a unitary summary and ordered detail panels.
"detail" forces panels; "overview" omits them. gates_per_panel controls the
instruction count (default eight). TeX measures each panel: two panels share a
row when both fit half-width, otherwise they stack at unchanged font size.
Registers continue between panels without resets; explain this in the caption.
Terminal IR annotations appear once. All modes validate every circuit operation.
The double line denotes declared classical readout values after the readout.

Rendering rejects panels wider than the canvas and complete figures higher than
170 mm rather than shrinking text. Reduce gates_per_panel, shorten labels, use
summary strings for long classical chains, choose overview or split circuit
detail into separate figures. This is a linear encoder/circuit/head description,
not arbitrary PyTorch tracing or branching-graph execution.

Save the PDF and include it in a 183 mm double-column article using `figure*`
and `\includegraphics[width=\textwidth]{hybrid.pdf}`. Do not shrink the wide
asset to a single column. The Notebook contains a matching caption and inclusion
snippet. Long unrolled circuits normally belong in supplementary material;
source remains editable for venue-specific composition.

Run the fully executed [Notebook example](../../examples/notebooks/hybrid_latex_drawing.ipynb)
for a real CPU CNN -> parameterized three-qubit circuit -> two-output head,
actual shape checks, joint and frozen-encoder training, five inline figures,
ordered long-circuit panels and portable export. The
[layout validation record](../../docs/development/HYBRID_LAYOUT_VALIDATION.md)
records test scope and PDF visual review.

## Limits, licenses and contributor checks

Noise channels, explicit matrices, dynamic control, and non-scalar batched gate
parameters are rejected until dedicated rendering rules exist. Observable and
terminal measurement requests are annotated without inventing state collapse.
Labels and symbolic names are escaped rather than treated as arbitrary TeX.
User-provided `LatexFigure(source)` source must be trusted by its caller.

The default wheel/sdist contains yquant (LPPL-1.3c), PlotNeuralNet (MIT) and the
composition layer (MIT), with licenses/provenance under `tex/`. Python integration
is Apache-2.0. There are no new required Python dependencies. The
[API proposal](../../docs/api-changes/FQ-DRAWER-LATEX-20261021.md) records ownership,
compatibility, replacement and exit criteria for these presentation resources.

Run `tests/unit/test_drawer_latex.py`, `tests/test_drawer_ir.py` and
`tests/unit/test_drawer_input_ownership.py` after drawer changes. Optional real
compiler/image conformance is in `tests/integration/test_drawer_latex_rendering.py`.
Use Matplotlib's Agg backend for headless tests and close caller-owned figures.


The private `_hybrid_layout.py` owns tensor glyphs, aligned stage geometry,
measured circuit rows and font/width handling. `hybrid_drawer.py` owns input
validation and composes that renderer. The owned flaghybrid.sty supplies separate
TeX box registers and a standard Helvetica-compatible font dependency; upstream
yquant/PlotNeuralNet sources and licenses remain unmodified.
