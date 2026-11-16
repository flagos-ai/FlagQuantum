# Quantum lattice streaming and reflection

All 19 source stages, the original 8 by 8 lattice, four velocity magnitudes, nine CFL events and thirteen streaming and reflection updates. Take 2048 samples and retain both spatial and velocity marginals.

Run from the repository root with `python -m examples.synthesis.applications.cfd.qlbm.overview.main`. Open the [Notebook](walkthrough.ipynb) for the CFL schedule, sampled marginals and obstacle diagram.

Use the approved normalization of the four equal velocity amplitudes from 0.25 to 0.5. Recompute the CFL schedule with exact rational event times and bind it to the native quantum program. Keep the source field order, modulo-eight streaming and post-stream reflection. Use a local sample seed instead of the hosted random stream.

[Source provenance](provenance.json).
