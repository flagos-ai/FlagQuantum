# Source mapping and acceptance

Cells 2/4/5/7 preserve the complete resource-vulnerability incidence graph, the
illustrative attack chain, bipartite views and the resource conflict graph.
Choose the resource side explicitly rather than relying on unordered set
orientation. Deterministic force layouts replace Kamada-Kawai where used.

Cells 9/11/12 model minimum vertex cover and display the original problem.
Evaluate all 256 assignments, retaining the cost of one per selected resource
and penalty 10 per uncovered conflict edge. Direct Python model tables replace
Pyomo objects. Independent enumeration establishes 21 feasible assignments and
minimum size five; it is a labeled classical reference, not a quantum result.

Cells 14/16/20 use the existing native qfunc. Preserve eight resource bits,
three cost/mixer layers, original phase sign, all graph edges and no workspace.
Bind two nonzero parameter sets and compare every absolute complex amplitude
to an independent Circuit, including bit order and clean workspace. Report
actual resources from the program.

Cells 23/25 run the full 60-iteration optimization budget using CVaR 0.9 and
plot actual objective evaluations. Use native Nelder-Mead with the explicit
shared initial schedule in place of unspecified hosted optimizer details.
Record evaluations, selected parameters, cost and convergence status.

Cells 27/29 draw 1,000 actual seeded observations from each of the selected
quantum distribution and the uniform classical baseline. Do not filter
infeasible observations out of the count histogram. The shared subset example
ranks large subsets first; patching must instead rank feasible covers by
**increasing** size. Test the actual workflow to protect this distinction.

Cells 31/32 choose and display the best feasible observed cover. Cell 34
removes those observed nodes from the complete attack graph. Intersect the
bipartite resource side with surviving nodes before drawing. Check that every
vulnerability now has at most one resource neighbor, so no shared-vulnerability
resource pair remains. If no feasible cover was observed, report that outcome
without claiming a patched graph.

Before coverage credit, execute all seventeen mapped cells in a fresh kernel,
inspect all eight rendered figures, verify independent Circuit states for both
nonzero bindings and final selected parameters, validate actual shot counts,
minimum-cover ranking, resource refusals and unchanged parent identity. Run
strict quality gates, the frozen 216+6 replay and the complete local default
tier with frozen Python inputs, comparing its exact failing nodes to the
latest accepted baseline.
