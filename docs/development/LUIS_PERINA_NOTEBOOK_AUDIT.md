# Complete Luis–Perina phase-window notebook

Q-0623 maps all 19 original cells of
`phase_estimation/lp_resource_state.ipynb`, frozen blob
`5e4568579be1cbfec1c0ab614fb9bc98c9641590` at Qualtran v0.7.0 commit
`bcb17bb25360ed68ce3bc084645db3e929799389`. The native notebook is
`examples/notebooks/luis_perina_phase_window.ipynb`.

Every original cell retains its ordinal in metadata. Scientific formulas and
references are preserved. Source class names map to native domain names. The
three original factory values remain interim(5), resource(5), and resource(n).
Both one-level call graphs, their exact child arguments and weights, and all
three register signatures are compared against execution of the untouched
original notebook. The signature includes output-only fixed-point semantics.

The shared signature renderer handles concrete diagrams. The symbolic example
uses the shared graph annotation table to preserve width, side, encoding and
fractional width as declaration metadata. This presentation has no physical
wire-layout claim. No new renderer or execution path is introduced.

Two added cells show ordinary `circuit.append(op)` and the existing experimental
dynamic execution entry point. The bounded two-bit preparation verifies the
complete sine amplitudes and clean work after a seeded measured trajectory.
The all-history complex-amplitude and exact-inverse certificate is reused from
`LUIS_PERINA_STATE_FILE_AUDIT.md`. The added example reports actual primitive
inventory separately from the independent source QEC graph. It uses six clean
work wires and three measurement slots. Its model reports three And, 49 Clifford,
18 rotation and three measurement, while the emitted inventory includes the
actual rotation decomposition, predicate gates and reset operations.

The complete original and native notebooks execute in the same test. Native
execution rejects Qualtran, Cirq and OpenFermion imports. Five diagrams are
rendered and visually checked. The parent source adjoint, strict-type and
SDK-default synthesis gaps remain on Q-0625. This notebook does not claim to
resolve them. No source fixture was modified.

The batch regression reports 311 passed. Strict mypy covers the notebook test
and every extracted native code cell. Black, Ruff, Algorithms scope,
architecture, multi-level IR and protected API checks pass. Logs use
`g2-lp-notebook-` in the reference workspace. Rendered SVG/PNG evidence includes
the five diagrams and their contact sheet.
