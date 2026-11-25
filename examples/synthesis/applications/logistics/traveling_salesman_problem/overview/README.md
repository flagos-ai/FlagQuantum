# Traveling salesman with native QAOA

All 17 source stages, four cities, 16 assignment bits, eight QAOA layers, 150 evaluations of the lower 60 percent sampled cost, and 2,048 draws for each optimization, final and uniform histogram.

Run from the repository root with `python -m examples.synthesis.applications.logistics.traveling_salesman_problem.overview.main`. Open the [Notebook](walkthrough.ipynb) for the actual sampled path and exact comparison.

Use native COBYLA and the grouped pi-ramp schedule from the available Classiq SDK reference. Preserve the last evaluated native binding and use independent local seeds for optimized and uniform readout. Enumerate the 24 feasible open paths as the classical reference. Report any invalid observed assignment as sampled.

[Source provenance](provenance.json).
