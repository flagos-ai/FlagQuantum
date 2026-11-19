# Complete matrix-QPE workflow

The frozen quantum counterpart in the parent directory is unchanged. This
separate workflow migrates all nineteen original Notebook code cells:

| Original cells | Native workflow |
| --- | --- |
| 1,5,8,12 | Imports, normalization, precision and static per-power schedule |
| 15,17,19,20,22,24 | Original matrix, seed-8 amplitudes, ordered Pauli decomposition, dimensions and normalized operator |
| 26 | Native exact and approximate qfuncs, synthesis and independent full-state checks |
| 27,30 | Actual resource/depth reports and local text drawing |
| 33,35,36,38 | Two 10000-shot histograms, top-two energies and source resolution intervals |
| 40,41 | All four classical eigenvalues and sampled comparison plot |

The exact unitary uses a real-symmetric eigendecomposition, replacing SciPy
without adding a dependency. `qpe(lambda: unitary(...), phase_result)` retains
ordinary matrix-power semantics. The approximate route uses
`qpe(lambda power: suzuki_trotter(..., repetitions[power], ...), phase_result)`.
The compiler specializes the static powers, controlling each family member once.
The original repetitions 2,3,5,7,11,16,23 are preserved; repeating one approximate
unitary would implement a different algorithm.

All 512 absolute complex amplitudes, including the retained private environment,
are compared at tolerance 2e-11. The reference uses independent dense controlled
products and inverse Fourier; Pauli factors use cos(a)I+i*sin(a)P. This avoids
small-angle general matrix-exponential error in the reference. No phase alignment
or relaxed precision bound is used. Additional signed nonzero preparations,
all 128 signed output labels, both complete finite-shot histograms, source
intervals, original matrix reconstruction and plot data are tested.

The source's remote sampler was unseeded. Explicit seeds 1400/1401 make the
local 10000-shot workflows reproducible without claiming identical hosted
outcomes. Gray windows have width equal to the source energy resolution;
displayed result intervals use plus/minus that resolution, as in the source.
Neither convention repairs the deliberately approximate evolution's bias.

Run:

```sh
python -m pytest tests/integration/test_synthesis_matrix_qpe_notebook.py tests/integration/test_synthesis_native_qpe_power.py -q
```
