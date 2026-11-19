# Link-monitoring acceptance

Complete original 17-cell wireless sensor link-monitoring workflow: all 12 resources, 13 edges, node 1 fixed to one, one QAOA layer, penalty 2, CVaR 0.9, native 60-iteration search, actual 1,000 draws, observed minimum feasible cover and original/solution network views.

The source fixes resource node 1 in its classical model before synthesis.
Its captured quantum export leaves this node free, so the complete native
Notebook uses a separate qfunc: prepare node 1 with X, prepare the other eleven
nodes uniformly, retain the full cover cost and mix only the free nodes. All
twelve logical nodes remain explicit. The previously verified 216 quantum
counterparts, including the unfixed export, stay unchanged.

Direct Python cover cost and exhaustive constraint diagnostics replace Pyomo.
Penalty 2 matches all 21 captured Observable coefficients for all 4,096 words;
constant phase, bit order and fixed-node invariant are independently checked.
Native Nelder-Mead, explicit initial angles (0.13, -0.27), exact local CVaR 0.9
and 1,000 seeded actual draws replace implicit provider optimizer/session
choices. Preserve the 60-iteration budget and report real iterations,
evaluations and convergence. No hosted trajectory equivalence is claimed.
Select the smallest observed feasible cover; retain infeasible samples and do
not inject an exhaustively known answer. Spring layout replaces Kamada-Kawai
without adding SciPy. No remote jobs, release, hardware or advantage claim.

Enumerate all 4,096 costs independently from the 13-edge graph and compare
against the captured 21-term Observable. Require fixed node 1 for feasibility:
318 feasible assignments, classical minimum size five. Compare two nonzero
parameter bindings and the actual selected parameters with independent Circuit
absolute amplitudes, including global phase, bit order, clean workspace and
fixed-node invariance. A regression must reject the unfixed quantum export
when used as this complete Notebook. Verify actual optimizer trace, CVaR,
sample counts, observed minimum ranking, no injected optimum and width, memory
and operation refusals. Fresh-kernel execution inspects all four actual plots
and the Circuit diagram. Replay all 216 originals and six approved alternatives
before freezing Python inputs for the complete local default test tier.
Coverage credit follows terminal acceptance.
