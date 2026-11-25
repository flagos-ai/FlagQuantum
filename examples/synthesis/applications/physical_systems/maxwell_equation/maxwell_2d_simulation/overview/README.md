# Full-grid Maxwell evolution

Run `python -m examples.synthesis.applications.physical_systems.maxwell_equation.maxwell_2d_simulation.overview.main`, or execute `walkthrough.ipynb`.

The complete original problem uses a 32 by 32 spatial grid, four electromagnetic components, seventeen physical bits, time five and a rectangular conductor with bounds [9,24) on both axes. The native finite-difference operator has been checked against all 16,777,216 original matrix entries. Complete coherent tests also verify the borrowed work bit, block projection, anti-Hermitian encoding and controlled relative phase.

The effective evolution time is 640. All 1,373 fresh phase rows and 1,372 walk selections execute through public native Circuit operations. Reusing the complete seventeen-bit state between blocks bounds execution storage. Actual executed instruction counts include the complete sequence and use the native static-instruction basis.

The independent classical reference evaluates the same exponential through the existing native matrix-free exponential action in bounded Krylov windows. It reports its measured defect. Acceptance checks the original fidelity threshold, the actual projected scale of 0.5 and the absolute complex projection error. Field displays preserve the source global-phase alignment and per-component normalization.

The workflow extends the existing shared example Laurent factorization to the required bounded degree while retaining its original degree-33 default and all numerical guards. The source grid, time, initial Gaussian, geometry and full phase sequence are retained. Full CPU execution takes approximately half an hour on the current machine.
