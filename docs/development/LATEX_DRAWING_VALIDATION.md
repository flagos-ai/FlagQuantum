# LaTeX drawing validation

Date: 2026-10-21. This is local development evidence, not a release or hardware
scalability certification. The feature was developed from `39f1073b8` and rebased
onto `01280756b`; the intervening three main commits only change CPU CI policy,
not the tested drawing or simulation implementations.

## Verified user workflow

```python
import flagquantum as fq
from flagquantum.drawer import draw_hybrid

circuit = fq.Circuit(2).h(0).cx(0, 1)
assert isinstance(circuit.draw(), str)
figure = circuit.draw(format="latex")
assert "cnot q[1] | q[0];" in figure.source
figure.export_bundle("overleaf-circuit")
figure  # Notebook final expression compiles and displays an image
```

Image rendering needs a local TeX compiler and Poppler on the kernel PATH.
Source export and portable bundles require only the default FlagQuantum install.
See the [executed Notebook](../../examples/notebooks/hybrid_latex_drawing.ipynb)
and [drawer dependency matrix](../../flagquantum/drawer/README.md).

## Focused checks

- **54 passed**: new drawer unit tests, optional real-rendering integration tests,
  existing Circuit/IR text/Matplotlib tests, presentation ownership tests and
  executable public docstring examples.
- The real compiler gallery includes every registered unitary opcode. It caught
  and corrected the assumption that yquant has native `s` and `t` commands;
  those operations are emitted as standard labeled boxes instead.
- **98% statement coverage** across exactly the three new Python modules
  (`hybrid_drawer`, `latex_drawer`, `latex_figure`): 287 statements, 5 not covered.
  This is not whole-repository coverage. PyTorch was imported before starting
  pytest-cov because tracing its native extension import segfaulted in this
  environment; all new drawer modules were imported after instrumentation began.
- Black and Ruff pass on changed Python files. Strict mypy passes for the three
  new modules with imported dependencies followed silently and missing optional
  import stubs ignored. This is not a whole-repository typing certification.
- `tools/public_api_snapshot.py` passes without regenerating any API snapshot.
- Compiler path-scope and integration packaging scope checks pass.

To reproduce the focused run with real rendering tools available:

```bash
python -m pytest tests/unit/test_drawer_latex.py \
  tests/integration/test_drawer_latex_rendering.py \
  tests/test_drawer_ir.py tests/unit/test_drawer_input_ownership.py \
  tests/api_contract/test_public_docstring_examples.py -q
```

The rendering tests explicitly skip when their external tools are absent; the
measured run above had Tectonic 0.17.0 and Poppler available and did not skip them.
PDF, PNG and actual Notebook MIME display were exercised. SVG conversion command,
cache and error behavior were tested with an isolated converter fake; this host
has no pdftocairo, so real SVG conversion is not claimed here.

## Executed Notebook

The Notebook was executed with a real local Jupyter kernel, CPython 3.12.14 and
PyTorch 2.13.0 on CPU. Three output cells contain images: existing Matplotlib,
LaTeX circuit, and explicit CNN/quantum/head architecture. There are no error
outputs. The final hybrid PNG was visually inspected for label truncation,
readable circuit detail, arrow crossings and clipping.

The real synthetic training step reports:

| Quantity | Value |
| --- | --- |
| CNN gradient norm | 0.003033 |
| Quantum gradient norm | 0.001278 |
| Head gradient norm | 0.209511 |
| Cross-entropy loss | 0.707551 |

Gradients are finite and nonzero for all three stages, and an Adam step changes
the quantum parameters. These are correctness checks for this tiny CPU example,
not an accuracy or performance claim. The symbolic drawing matches its gate
sequence; the classical architecture is explicitly supplied, not traced.

## Installed distribution

A wheel and sdist were built using the declared PyTorch 2.10 build dependency.
The wheel's four drawer Python modules match the reviewed checkout byte for
byte. Both archives include yquant, PlotNeuralNet and component licenses; wheel
metadata records `Apache-2.0 AND MIT AND LPPL-1.3c`. Upstream runtime bytes,
including original line endings, are retained through local Git attributes.

The wheel was installed into a separate directory, and source generation plus
portable bundle export succeeded outside the repository. The installed S/T gate
rules and the included runtime/license paths were checked explicitly. This does
not publish a package to PyPI or certify the existing native extension on every
supported platform.

## Broader repository tiers

- `pr-runtime`: **1717 passed, 123 skipped**, 15367 deselected, 201.56 seconds.
  This run allowed the local loopback sockets required by existing single-rank
  MPS fixtures. Its new real-rendering tests ran and passed.
- `pr-default`: **3333 passed, 50 skipped**, then stopped at
  `tests/team/simulation/test_controlled_phase_graph.py::test_qft_hadamard_phase_fusion_matches_graph_path_and_reduces_passes[dtype0]`.
  It observes 44 statevector applies while asserting 28. The exact failure was
  reproduced from an untouched archive of the `39f1073b8` baseline in a separate
  directory using the same interpreter. No simulation implementation or
  assertion was changed to make this drawing feature pass. The default tier is
  therefore not recorded as green.

Normal maintainer review and required remote CI remain necessary before merge.
