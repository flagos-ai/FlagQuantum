# Decoded quantum interferometry for Max-XORSAT

All 15 source stages, the 15-bit unary example, both six-bit Dicke states, the six-cycle Max-XORSAT graph and all 22 lookup entries. Use 2,048 shots for each preparation and 10,000 DQI samples.

Run from the repository root with `python -m examples.synthesis.algorithms.search_and_optimization.dqi.dqi_max_xorsat.overview.main`. The [Notebook](walkthrough.ipynb) shows the source graph and measured score distribution.

Prepare the unary example from its known classical count and use a reversible preparation specialized to the specified unary basis input for the second Dicke example. Regenerate the incidence and decoder tables and solve the original three-by-three eigenproblem. The native circuit keeps both output words and tests the zero-error word directly from measured results.

[Source provenance](provenance.json).
