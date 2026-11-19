# Complete THC physical compilation tutorial

Q-0929 maps all seven cells of the frozen
`qualtran/surface_code/thc_compilation.ipynb` at Qualtran v0.7.0 revision
`bcb17bb25360ed68ce3bc084645db3e929799389`, blob
`e82d0fee44ceb5f9cf2216ba929358e140003869`, into
`examples/notebooks/thc_physical_compilation.ipynb`.

The notebook retains both workflows, source cell order and source-cell indices.
A final cell records provenance and scientific scope. The manual model uses
6,665,400,000 Toffolis, 696 logical qubits, four factories with distances 19/31,
data distance 31 and routing overhead 0.5. The deprecated source wrapper maps
directly to the shared native PhysicalCostModel. No redundant helper is added.

The complete default odd-distance factory/layout search uses four factories,
error budget 0.01 and minimum duration. The selected distances are 17/29 for
factory levels and 33 for data. The exact result is failure probability
0.007725395132201774, 2,933,032 physical qubits and 89.1034375 hours.
The unmodified reference notebook executes successfully. Every native code cell
executes with Qualtran/Cirq/OpenFermion imports blocked. Manual and optimized
cost fields, qubit-hours, both selected factory/data configurations, separate
factory/data errors and all printed output match exactly.

The source's explanation of disagreement with the THC paper is retained. Its
50% routing overhead and factory-limited timing approximate detailed lattice
surgery. The tutorial consumes a declared workload and does not compile or
simulate the FeMoco Hamiltonian. Primitive emission, complex amplitudes,
measurement histories and workspace cleanup remain owned by the emitting
components. The cost notebook introduces no new emitted circuit.

All direct dependencies already have complete-file acceptance. The retained
one-pass data-layout iterator defect is documented by the search-file audit.
This tutorial uses the source reusable default layout tuple, so its full
candidate search remains covered. Source strict-budget rounding and unrelated
component-diagram boundaries remain separate parent items.

All 446 concentrated resource regressions pass. The notebook test and extracted
script pass strict mypy, Black and Ruff. Actual code cells are individually
Black checked. Core scope, architecture, IR and protected API checks pass.
