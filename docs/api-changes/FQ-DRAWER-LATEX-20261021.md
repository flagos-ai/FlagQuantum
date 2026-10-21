# LaTeX circuit export and Notebook display

## Decision and authorization

The user explicitly requested preserving existing draw behavior while adding
LaTeX source export, direct Notebook display and a quantum-classical model
Notebook on 2026-10-21. This authorizes this additive drawer backend and its
namespace APIs; it does not authorize changes to Core IR or simulation semantics.
Owner: drawer/compiler; packaging is an integration surface. Normal maintainer
review and CI are still required before merge. No protected API snapshot is
regenerated as an approval mechanism.

## User journey and interface

`circuit.draw()` still returns text. `circuit.draw(format="mpl")` still returns
`(Figure, Axes)`. `circuit.draw(format="latex")` returns `LatexFigure`, with
`.source`, `str(figure)`, `.save_source(path)`, `.export_bundle(directory)`,
`.render(format="pdf"|"svg"|"png")`, `.save(path)` and the Notebook MIME protocol.
`standalone=False` makes source embeddable; rendering wraps that fragment.
`display=True` explicitly displays the same object and returns it.

The `flagquantum.drawer` namespace exports `LatexFigure`, `draw_latex` and
`draw_hybrid`. Nothing is added to the stable root namespace. The existing draw
signature, defaults and text/mpl behavior stay intact; only its supported formats
and return union grow. Unknown formats still raise ValueError.

`draw_hybrid(circuit, ...)` renders the concrete encoder -> quantum -> head
example, optionally with loss/optimizer and three parameter updates. Its labels
and feature counts are supplied explicitly. It does not trace a PyTorch model,
execute a forward pass or introduce a second executable IR. Existing compiler
HybridProgram is an execution abstraction and does not describe CNN volume
geometry or layout, so it is not repurposed for presentation.

## Dependencies and ownership

TeX source export uses Python's standard library and the existing Core operator
schema and parameter types. TeX runtime resources ship in the default wheel and
sdist. A user's installation need not clone a second repository. yquant files
remain unmodified under LPPL-1.3c, PlotNeuralNet under MIT, and the wrapper retains
its MIT notice. Distribution metadata includes these component licenses.

Rendering is optional and external: Tectonic, or a supported TeX Live engine,
compiles PDF. Poppler pdftocairo produces SVG; pdftoppm (or pdftocairo) produces
PNG. Notebook auto-display prefers SVG when pdftocairo exists and otherwise uses
PNG. No compiler or converter is downloaded or installed by library code.
IPython is imported only for an explicit display call. There is no new required
Python dependency. The existing `viz` extra adds BSD-licensed IPython for explicit
display and formatter conformance; it stays lazily imported and is replaceable
at the MIME/display boundary. There is no Matplotlib requirement for source export or network
or subprocess work at import or source-generation time.

The drawer owns the resource reader and bounded compiler/converter boundary.
Resources are copied into an isolated temporary build directory, subprocesses
run without a shell and with timeouts, and TeX shell escape is disabled.
Rendered bytes are cached only on the figure instance with immutable source;
there is no persistent global cache. Missing tools, unsupported semantics and
compile errors are actionable failures, never substituted with another circuit.
The replacement interface is source -> rendered bytes; removing the vendored
runtime in a future release requires replacing the resource bundle and passing
source, installed-wheel and rendering conformance tests. No simulator consumes
these resources.

## Compatibility and limits

No text or Matplotlib migration is necessary. Existing Circuit/CircuitIR gate
semantics are read from Core. Noise channels, explicit matrices and dynamic
control are rejected by this backend until their rendering rules are supported.
Terminal readout requests and observables are annotations, not invented
collapse operations. Counts and labels never modify model parameters or gradients.
The first hybrid renderer is an explicit three-stage layout, not general graph
tracing. The Notebook demonstrates a real CPU PyTorch CNN/quantum/head forward
and backward pass and separately declares the matching drawing.

## Verification

Required evidence: legacy drawer tests, native Circuit/CircuitIR parity,
symbolic/control/unsupported-operation cases, source/gradient ownership,
fragment compilation, missing-tool/compiler/converter/timeout failures, cache
reuse, wheel/sdist resource inclusion and source generation outside the checkout,
plus a fully executed Notebook with an actual image MIME output. Targeted
formatting, lint/type checks and the repository's default and runtime test tiers
are recorded with their measured scope; unrelated failures are reported intact.
