# Network resiliency planning

Run `python -m examples.synthesis.applications.telecom.resiliency_planning.overview.main` from the repository root, or execute `walkthrough.ipynb`.

All 21 original stages retain five nodes, seven edges, two demands, 24 assignment bits and twenty QAOA layers. The frozen native qfunc builds the full cost and RX computation from the original forty-angle schedule. Independent local factor tables validate all quadratic coefficients and two full complex statevectors, including absolute global phase.

The native optimizer performs one actual objective query and an independent final sample. Both use 20,000 real CPU draws. The original Notebook declares this shot count without passing it to its hosted primitives. Making it explicit is a local execution choice. The query cap reports a budget stop and unconverged status. No historical optimizer trajectory is claimed.

Validity follows the source flow and node-overlap checks exactly. The correlated-edge term stays in the objective. The illustrative assignment appears separately from sampled routes. Only valid outcomes in the actual top 100 are plotted, and reported probabilities use the actual final shot count. Native static resources have their own basis.
