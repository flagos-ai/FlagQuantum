# Complete Beverland literature notebook

Q-0903 maps all 57 cells of the frozen
`qualtran/surface_code/beverland_et_al_model.ipynb` at Qualtran v0.7.0 revision
`bcb17bb25360ed68ce3bc084645db3e929799389`, blob
`1dca78647d0fd0f8d0a8cd8e37c64861b066fbc0`, into
`examples/notebooks/beverland_resource_model.ipynb`.

Each cell retains its original order, type and source-cell index. One additional
cell records copyright, license and numerical interpretation. Native execution
uses existing logical resources and domain-owned hardware, data-layout,
rotation-cost and code-error models. Rotation layers belong to the existing
minimum-time consumer argument. No duplicate algorithm-summary type is added.
Integral floating-point gate literals become exact integers.

Every code cell is executed alongside the immutable original. Native execution
blocks Qualtran, Cirq and OpenFermion imports. All printed results and all scalar
intermediates after every code cell match exactly, including logical tiles,
rotation synthesis demand, minimum time, distances, duration, factory footprints
and total physical qubits. All three input workloads are compared independently.
The source notebook was already executed unchanged in an isolated kernel during
the complete Beverland-formula batch. That evidence is reused.

The source text quotes an affine intercept of 4.86 while its executable rotation
model uses 5.3. Both are retained and the discrepancy is explicitly explained.
Dynamics uses 140000 measurements. Factoring uses 3730000000 Toffolis. The
source's suspected-paper-typo assumptions remain visible. Fixed factory counts,
footprints and rates are literature inputs, not newly synthesized factories.

This notebook estimates scalar logical and physical resources. It emits no
primitive circuit. Amplitudes, measurement branches and workspace cleanup are
outside this notebook's domain. Existing circuit construction retains
`circuit.append(op)`. Q-0924 strict-budget rounding and Q-0613 component-diagram
boundaries remain separately open.

All 390 concentrated resource regressions pass. The notebook test and extracted
script pass strict mypy, Black and Ruff. Actual cells are individually Black
checked. Core scope, architecture, IR and protected API checks pass.
