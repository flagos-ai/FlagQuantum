# Native quantum-volume source experiment

Run from the repository root:

```bash
python -m examples.synthesis.applications.benchmarking.quantum_volume.overview.main
```

Open `walkthrough.ipynb` for all eleven source code stages, including the four
nonexecuting hardware configurations. `ansatz.py` generates fresh random
matrices and composes captured native quantum functions. `workflow.py` runs all
forty original-sized circuits with 100 draws each, independent Circuit and
integer-basis references, exact absolute phase checks and named marginal checks.
All forty synthesized programs remain available in `experiment.programs`.
`plots.py` compares the original observed-count score with ideal heavy mass.

Array element zero is the least significant bit. A two-element target tuple
uses its first element as the local matrix least significant bit. Both reference
paths implement these conventions explicitly; neither calls the native lowering.

The source score takes the median of positive observed counts, includes ties,
and divides the population deviation of heavy counts by 1000. Its first-failure
threshold is retained exactly. This differs from standard ideal-median heavy
output generation; the printed source volume is a diagnostic, not certification.
The plot and output keep those two statistics separate.

Fresh Haar factors use NumPy Gaussian QR with the same unitary distribution as
the source SciPy factors. Both seeded RNGs start at zero; exact SciPy random
sequence identity is not asserted. No new dependency, API or compiler change is
needed. The original frozen Qmod counterpart in the parent directory remains
unchanged. No source loading, provider execution, noise model, hardware, speedup
or scalability claim is included.
