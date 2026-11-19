# Source-sized native HHL workflow

This is a `single_device_fast_path` example. Quantum construction uses public
`flagquantum.synth` preparation, QPE, matrix unitary, selected Y rotations and
`within()`/`apply()`. The inverse QPE is generated automatically.

The original matrix entries and vector [1, 2, 4, 3] are computed afresh with
PyTorch, which is already a dependency. NumPy/SciPy matrix exponential, solve
and norms become their PyTorch equivalents. The source fixed matrix callback
is expressed directly in the native root; no generic QCallable API is claimed.
The original fresh-output preparation helper is run as a root for its exercise
and its preparation statement is expressed directly in the complete root.
The borrowed reciprocal helper receives an indicator allocated before the
reversible region, so no allocation occurs inside the unitary computation.

The full reference Circuit uses independent Householder preparation, controlled
matrix powers, dense Fourier transforms and reciprocal table blocks. Validation
compares every complex amplitude without global-phase alignment. It separately
checks compiler work and the retained physical phase register. Finite QPE
precision produces nonzero phase leakage on indicator=1 branches; it is reported
and is not called dirty compiler work. No phase register is dropped or reset.

Actual source postselection selects indicator=1 and phase_=0, keeps amplitudes
above 1e-10, divides by 1/16 and rounds both complex parts to five decimals.
The source removes a separate phase from each component in its final diagnostic.
That computation and plot are preserved explicitly as a magnitude diagnostic,
not a physical global-phase correction or proof of state equivalence.
The complex solution fidelity and relative distance are reported separately.

Resource reports describe synthesized instructions before Runtime compilation;
they replace the hosted viewer without claiming CX/U-basis equivalence.
No source importer, compiler/API/dependency change, hardware run, speedup,
scalability or resource optimality is claimed. The frozen quantum counterpart
in the parent directory is unchanged.
