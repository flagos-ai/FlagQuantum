# Four-stage workflow acceptance

Retain all source quantum and classical stages: seed 123 amplitude generation,
normalization, Type I blocks, combined Type II selector blocks, standalone
Type II cosine and sine, execution, readout and printed measured/expected arrays.
Source code cells are reviewed and hash-bound in provenance. No original source
code or importer is included in user examples. The source has no plots or live
hardware execution.

Native qfunc uses existing prepare_amplitudes, qct_qst_type1, qct_qst_type2,
qct_type2 and qst_type2. There is no new production code or dependency. NumPy
retains the original RandomState sequence; Torch supplies classical references.

Independent Circuit comparison reuses the existing Fourier reference helper,
which prepares each input with a Householder reflection. Type I uses a complex
Fourier matrix in an even/odd parity basis. Type II uses real/imaginary Fourier
coefficients, independently of the compiler and the native example's direct
cosine/sine formulas. Full absolute complex amplitudes must agree at 1e-12, with
no phase alignment. Check block signs, logical register order and zero work.

The four stages have widths 4, 4, 3, 3 and workspace 0. Type I's sine block has
relative phase i; combined Type II's sine block has a minus sign. Zero-work
acceptance is possible because these transforms introduce no hidden outputs or
environment bits. Explicit width, memory, gate-count and depth ceilings must
refuse infeasible runs. Classical references reject invalid sizes/types instead
of allocating unbounded dense matrices. Notebook code cells are checked by
execution and typing; actual Jupyter and both CLI launchers complement tests.

```sh
python -m pytest tests/integration/test_synthesis_qct_qst_notebook.py -q
```
