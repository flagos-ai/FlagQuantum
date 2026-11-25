# Source workflow mapping and acceptance

Pinned source: `applications/optimization/max_k_vertex_cover/max_k_vertex_cover.ipynb`.
`provenance.json` records all 16 source code cells and their hashes.

| Source code cells | Native workflow |
| --- | --- |
| 2, 4, 6, 7 | Original ten-node Erdos-Renyi graph, edge probability0.5 andseed13, K=5; independent uncovered-edge/cardinality table replaces Pyomo. |
| 9, 11 | Reuse the unchanged native cost helper and three-layer main, uniform preparation and RX mixers; inspect actual synthesis resources. |
| 13, 15 | Actual synthesis and execution on every optimization evaluation; CVaR0.7, 90-iteration setting and measured cost trace. |
| 18, 20, 22 | 1000 actual optimized and 1000 uniform CPU observations; penalized costs and common-bin40 histograms. |
| 24, 25 | Select the best actually observed feasible assignment; show its five nodes and covered edges on the original graph. |
| 28, 29, 30 | Separate bounded exact reference after sampling, with its own graph panel; replaces external Couenne. |

Let U(w) count edges with neither endpoint selected, and n(w) count selected
nodes. The raw objective is U(w), feasibility is n(w)=5, and the quantum cost is
C(w)=U(w)+10*(n(w)-5)**2. Each cost layer applies exp(-i*gamma*C(w)), including its
constant global phase, followed by RX(beta) mixers. Preserve the source graph's
21 edges, ten decision bits and all three layers. Angles are radians.

The frozen native quantum program and its Circuit counterpart remain unchanged.
The complete workflow reuses those quantum functions plus the existing named
readout, CVaR calculation and evaluation record. Its local sample ranking puts
feasibility first, then penalized cost and word; the older subset-size ranking
would not express this uncovered-edge objective. Keep feasibility, uncovered
edges and penalized cost as separate fields. No feasible observation stays None.

Public Nelder-Mead with simplex step0.07 and initial parameters
(0.13,-0.27,0.19,0.22,0.31,-0.17) replaces the source private optimizer. Exact local
probabilities replace shot estimation during the lower-tail CVaR0.7 search.
Every actual evaluation synthesizes, executes and reads the native program.
The90-iteration setting is preserved; iteration/evaluation semantics and optimizer
trajectories are not identical. Report the actual evaluation count and convergence.

Final CPU multinomial sampling uses seed2718 for the optimized state andseed31415
for the uniform baseline, 1000 observations each. A reference minimum is computed
from the252 feasible assignments only after optimization and sampling. It never
supplies parameters, probabilities or missing observations. Finite sample success
does not prove variational convergence, hardware parity or quantum advantage.

Node i is bit i of the named v integer (node0 is LSB). Physical statevector axes
are treated separately. A reproducible spring layout with seed2718 replaces
source Kamada-Kawai and avoids a new solver dependency. Preserve original graph,
cost trace,40-bin sample histogram,observed solution andclassical-reference graph
roles. Common bins make the two actual sampled distributions comparable.

Acceptance checks all1024 assignments against an independent graph objective,
all1024 absolute basis phases at two nonzero gammas, two complete complex states
against the independent Circuit, all ten named node positions,zero workspace,
feasibility boundaries, missing feasible observations, invalid controls and
width/depth/operations/memory budget refusals. It also executes the clean
Notebook in a fresh namespace and a real Jupyter kernel and inspects the figures.

Classification: `single_device_fast_path`. All execution is local; no remote
services, external optimizers or exact-solver packages are required.
