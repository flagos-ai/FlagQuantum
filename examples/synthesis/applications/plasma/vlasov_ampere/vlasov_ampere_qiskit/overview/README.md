# Vlasov-Ampere Circuit block-encoding experiment

Run `python -m examples.synthesis.applications.plasma.vlasov_ampere.vlasov_ampere_qiskit.overview.main`, or execute `walkthrough.ipynb`.

All 25 original stages retain three spatial bits, three signed velocity bits, the electric-sector bit, ten block/work bits and the full 128-component input. The source temperature, domain, boundary stencil, force/current coefficients and frequency remain intact. A fresh seeded local input is used because the source input is unseeded.

The public native Circuit composes controlled complex128 matrices for the original derivative, advection, boundary-selection and off-diagonal branches. The four-bit Draper adder becomes its exact modular permutation. Real amplitude preparation uses a declared Householder extension. This preserves the full projected operator and its absolute complex phase. Other unprojected amplitudes can depend on that extension.

The full 17-bit state executes before block-zero selection. Compare all 128 projected complex amplitudes against a fresh independently assembled classical matrix using the original L2 threshold of 1e-10. Both original work bits must return to zero. Two fresh input seeds are checked.

The final source example constructs a single QSVT demonstration pair with angles 0.1 and 0.2. Its native counterpart retains all 18 bits, both projector phases and U/adjoint composition. It has additional unitary and zero-angle roundtrip tests. The native resource report uses `static_instruction_sequence` and counts controlled matrix operations. Its values have a different basis from the source U/CX decomposition.
