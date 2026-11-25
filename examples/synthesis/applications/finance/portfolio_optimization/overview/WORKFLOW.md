# Source mapping and acceptance

Cells 2/4/6 define the three-asset data and integer-constrained portfolio model.
Retain both off-diagonal covariance contributions, the return signs, holdings
0..7 and the at-most-six budget. Direct Python objective and feasibility
functions replace Pyomo objects without changing the mathematical problem.

Cells 8/10 construct the original three-layer quantum program with penalty 10.
Reuse the frozen native parent and its existing bounded slack weights 1/2/3.
Report actual resources and verify two nonzero parameter sets against the
independent Circuit, including every complex amplitude and absolute phase.

Cells 12/14 run optimization and plot the trace. Retain the 60-iteration budget
and quantile 0.7. Native COBYLA consumes exact local encoded-cost probabilities;
its explicit initial schedule and trust radii replace implicit hosted details.
Label the trace by objective evaluations, not by an invented provider iteration
record. Report termination and the actual evaluation count.

Cells 18/20/22 sample optimized and uniform programs and compare their weighted
cost distributions. Use 1,000 actual seeded draws per distribution where the
source leaves the shot count unspecified. Marginalize slack using named
registers and report original cost separately from feasibility. Do not silently
discard infeasible outcomes. Cell 23 reports the best feasible observed holding;
report no feasible sample if none was drawn. Cell 25 replaces an external Couenne
solve with exhaustive enumeration of the entire 512-point classical domain.
This independently certifies the classical optimum without determining the
quantum sample result.

Acceptance must verify cost and slack encoding, fractional CVaR boundaries,
sampling counts/seeds, the original optimization budget, actual trace and final
state cost, independent full states, parameter/budget refusals and frozen-parent
identity. Inspect both rendered charts, execute all twelve mapped cells in a
fresh kernel, run strict quality and the frozen 216+6 replay, then compare the
required complete local default tier before granting Notebook coverage credit.
