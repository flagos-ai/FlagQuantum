# Complete physical-cost notebook

Q-0918 maps the entire frozen notebook
`qualtran/surface_code/physical_cost_model.ipynb` at Qualtran v0.7.0 revision
`bcb17bb25360ed68ce3bc084645db3e929799389`, blob
`5cb5070d0ea667fcb51b1fe972258d8761f8240d`, to
`examples/notebooks/physical_cost_model.ipynb`.

All original 21 cells retain their order, type and explicit source-cell index.
The additional final Markdown cell records provenance, license and numerical
limits. The notebook uses native logical resources, the shared physical model,
data layouts, factory interfaces, error scheme and hardware parameters. All 48
original combinations remain. Domain imports are explicit and do not add stable
exports or an upstream-branded namespace.

The execution test runs every source cell and every native cell. Imports of
Qualtran, Cirq and OpenFermion are blocked during native execution. Duration and
physical-qubit arrays, scatter coordinates, both default model outputs, both
data block examples and the billion-Toffoli factory example are compared.
All cycle/time/space comparisons are exact. Source asset hashes and cell mapping
are checked. The preceding 48-scenario model matrix records the maximum absolute
error difference 3.9968028886505635e-7, bounded by 8e-7 from 2e-15 per T state.

The scatter is rendered at 160 dpi and visually inspected. Axis labels remain
readable and all source coordinates are retained, including coincident points.
The original unmodified notebook test from Q-0920 already passes in an isolated
kernel and is reused. A strictly checked extracted script hoists only top-level
imports. Actual notebook cells are individually Black formatted and their imports
sorted without discarding names consumed in later cells.

The notebook describes physical scalar estimates. Primitive emissions and QEC
categories remain separate inputs. The original high-error example is retained
without clipping its first-order estimate. No new measurement histories or
workspace claims are introduced. Q-0924 strict budget rounding and Q-0613
component-diagram gaps remain independently open.

All 289 concentrated physical-resource regressions pass. The notebook test and
extracted script both pass strict mypy, Black and Ruff. Core-team scope,
architecture, IR and API checks pass. Q-0918 closes as a complete notebook asset.
