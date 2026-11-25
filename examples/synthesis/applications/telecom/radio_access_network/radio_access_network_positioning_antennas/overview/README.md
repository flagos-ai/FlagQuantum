# Native antenna placement

Run `python -m examples.synthesis.applications.telecom.radio_access_network.radio_access_network_positioning_antennas.overview.main`, or execute `walkthrough.ipynb` from the repository root.

All thirteen input coefficients are regenerated with the original NumPy legacy seed zero. The source objective maximizes their selected sum, despite calling them consumption coefficients. Preserve that sign, at most seven antennas, the three exact neighbor groups and all three modeled binary slack bits. Four native cost/mixer layers, penalty three, sixty mean-cost iterations and 3000 actual final CPU shots remain fixed.

Public native exact expectations and Nelder-Mead replace proprietary shot-based optimization. Each angle tuple is synthesized anew. Raw maximized score, source penalized cost and decision feasibility are reported separately, including infeasible observations. The existing parent Qmod-derived Hamiltonian has a different identity coefficient from the freshly generated Notebook model and stays frozen; it cannot replace fresh data here.

See [WORKFLOW.md](WORKFLOW.md) and [provenance.json](provenance.json). No private FlagQuantum API, source loader, frozen-data substitution, optimum guarantee, hardware, speedup or scalability claim is used.
