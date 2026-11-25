# Source workflow mapping

The pinned source is the complete two-section QAOA Notebook, including both
MaxCut and Knapsack. `provenance.json` records all 20 reviewed code cells.

| Source cells | Native workflow |
| --- | --- |
| 15, 16, 18 | Native imports and original five-node, six-edge graph; seeded spring layout. |
| 22, 24, 27, 30, 33 | Negative normalized cut objective, phase helper, RX mixers, four-layer qfunc and uniform preparation. |
| 35 | Synthesize with explicit five-bit, zero-workspace and operation/memory budgets; inspect actual resources. |
| 39, 41 | Execute native programs at every optimizer evaluation; retain the linear initial schedule and 60-iteration budget. |
| 43 | Plot the actual per-evaluation expected-cost trace. |
| 46, 48 | Draw 1000 final local samples, show top counts and every best observed cut sorted by count. |
| 56, 58, 60 | Three-bit a and two-bit b; value 3a+5b; coherent predicate 2a+3b <= 12; three cost/mixer layers. |
| 62 | Retain the pi-scaled initial linear schedule and 60-iteration budget; plot actual expected cost. |
| 64, 66 | Draw 1000 final local samples; print named a/b, objective, feasibility and count. |

Reuse the existing native quantum functions unchanged. Both independent Circuit
references verify complete complex amplitudes, including global phase and named
register order, at two nonzero parameter bindings. Native synthesis uses five
physical bits, zero workspace and no dense reference matrix.

The source uses SciPy COBYLA and execution sessions. This local implementation
uses the existing public Nelder-Mead optimizer with initial simplex step 0.07.
The source iteration setting remains 60, but iteration/evaluation semantics and
optimization trajectories are not identical. Every actual evaluation synthesizes,
executes and reads the named probability distribution. Exact local expected cost
replaces shot estimation during optimization; no quantum advantage is claimed.

MaxCut minimizes negative cut count divided by six. Knapsack minimizes -3a-5b
when 2a+3b <= 12 and zero otherwise. Its negative conditional phase is preserved;
infeasible assignments are separately flagged during sample ranking. The
Knapsack mixer uses commuting X evolutions equal to RX(beta), including phase.
Final CPU multinomial samples use seed 2718 and 1000 shots per section. A missing
feasible observation remains missing; no exact solution substitutes for it.

Bounded exact enumeration over 32 assignments is only an independent test
reference, calculated after optimization and sampling. Neither it nor the
independent Circuit supplies optimizer parameters or sampled answers.

Array node i is bit i of the named v integer. Named QNum a and b are read as
integers directly; physical statevector axes are treated independently. A seeded
spring layout replaces the source unseeded layout for reproducibility. The
original graph, cost trace and all best-sampled-solution graph roles remain.
Knapsack also shows a table of actual sampled assignments beside its cost trace.

Classification: `single_device_fast_path`. No cloud, hardware, foreign solver,
optimizer convergence or optimizer trajectory equivalence is claimed.
