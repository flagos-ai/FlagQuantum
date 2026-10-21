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
from flagquantum.drawer import draw_hybrid

figure = draw_hybrid(
    circuit,
    encoder="CNN encoder",
    encoder_features=2,
    head="Dense head",
    head_outputs=2,
    training=True,
)
figure
```

The architecture labels and feature counts are explicitly supplied presentation
data, not an inferred PyTorch graph. The embedded circuit is read from IR.
Training arrows illustrate data and parameter updates; drawing never invokes
forward or backward execution. Large circuits are best inspected with a separate
circuit figure because the hybrid view has a fixed-width circuit inset.

Run the fully executed [Notebook example](../../examples/notebooks/hybrid_latex_drawing.ipynb)
for a real CPU CNN -> parameterized three-qubit circuit -> two-output head, a
training step with gradients in all components, inline figures and portable export.

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
