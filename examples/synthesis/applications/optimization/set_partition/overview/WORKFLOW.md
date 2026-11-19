# Source workflow mapping and limits

Pinned source: `applications/optimization/set_partition/set_partition.ipynb`.
`provenance.json` records every reviewed code cell. This is the complete local
workflow, rather than only the frozen quantum counterpart.

| Source code cells | Native workflow |
| --- | --- |
| 2, 5 | Local NumPy RandomState(0), randint(1,12,10); preserve all ten source weights without mutating global RNG state. |
| 4, 6 | Independent classical squared group-sum difference table; no Pyomo dependency. |
| 8, 10 | Native parameterized cost qfunc, three negative-cost/RX layers, uniform preparation; inspect actual synthesis resources. |
| 12, 14 | Actual synthesis/run per evaluation; CVaR0.7, 80-iteration setting and cost trace. |
| 17, 19, 21 | 1000 final optimized and 1000 uniform CPU observations; ranked costs and common-bin weighted histograms. |
| 23, 24 | Best actually observed word, source item membership, group values and sums. |
| 26, 27, 28 | Separately labeled bounded exhaustive classical reference, computed after search and sampling. |

The source Notebook weights are (6,1,4,4,8,10,4,6,3,5), total51. The frozen
Qmod counterpart contains (4,9,3,9,2,6,9,10,7,8), total67. Do not substitute
that frozen dataset for the Notebook or modify its historical resource baseline.
Weights are bound as classical integer parameters; the cost kernel is reusable
for ten integer weights in [1,32]. The validation uses a second nonzero dataset.

For word w, cost=(sum_i weights[i]*(2*bit_i(w)-1))**2. Each cost layer applies
exp(-i*gamma*cost), including the constant global phase; mixers are RX(beta).
All assignments are feasible, so the source penalty factor10 adds no penalty.
Array v[i] denotes bit i of the named integer (item0 is LSB). Physical statevector
axes are handled separately. Selected means bit1 and other means bit0; source
P2 and P1 respectively. Duplicates remain separate indexed items.

Public native Nelder-Mead with simplex step0.07 and initial parameters
(0.13,-0.27,0.19,0.22,0.31,-0.17) replaces the source private optimizer. Each
actual evaluation synthesizes, executes and evaluates lower-tail CVaR0.7 from
exact local probabilities, replacing the shot estimator during optimization.
Retain 80 iterations but disclose actual evaluations and convergence; optimizer
trajectory, iteration semantics and hardware parity are not claimed.

Final CPU multinomial sampling uses seed2718 for the optimized distribution and
seed31415 for the uniform baseline, 1000 observations each. The best observed
row is selected from these observations alone. The exact1024-assignment search
replaces external Couenne and is calculated only after sampling; it does not
supply optimizer parameters, probabilities or observations. Report every tie's
count separately from total optimum-frequency. Finite sample success does not
prove variational convergence or quantum advantage.

Acceptance compares complete complex statevectors against an independent
spin-parity Circuit at two nonzero parameter bindings on both datasets, and all
1024 basis phases against direct classical group sums. No phase alignment is
allowed. Tests cover named bit order, duplicate weights, zero workspace, invalid
bindings and width/depth/operations/memory budget refusals. The four-panel figure
preserves actual cost trace, sampled histograms and group/reference roles.

Classification: `single_device_fast_path`. No remote service or external solver
is required; no cloud, hardware or convergence claim is made.
