# Rectangle packing with native QAOA

All 19 source stages, the 4 by 2 container and three rectangles, 17 active variables, 102 regenerated Ising terms, ten QAOA layers, 60 objective queries and 10,000 actual shots per query and final readout.

Run from the repository root with `python -m examples.synthesis.applications.optimization.rectangles_packing.rectangles_packing_grid.overview.main`. Open the [Notebook](walkthrough.ipynb) for the exact geometry, finite query trace, sampled costs and observed layout.

Native local COBYLA and the nonconstant-coefficient norm schedule reproduce the optimization setup at the original sizes. Finite sampling uses fixed local seeds. The final reported packing is a word observed in the final sample, and its cost is checked against the separate exact geometry table. Query traces describe the native run.

[Source provenance](provenance.json).
