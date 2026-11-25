# Source workflow mapping and acceptance

Pinned source: `applications/optimization/integer_linear_programming/integer_linear_programming.ipynb`
at `b69c4893d8264d1e59ce0909df2bd88efed25e00`.
`provenance.json` records every original code cell and its hash.

| Source cells | Native workflow |
| --- | --- |
| 3, 4, 5 | Original A=((1,1,1),(2,2,2),(3,3,3)), b=(1,2,3), c=(1,2,3), x in {0,1,2,3}^3. Check all three A*x<=b inequalities and maximize c*x. |
| 7, 9 | Reuse the frozen native three-layer quantum function and inspect actual synthesis resources. |
| 11, 13 | Actual synthesis and execution on every optimization evaluation; CVaR 0.7, 90-iteration setting and actual evaluation trace. |
| 17, 19, 21 | 1000 optimized and 1000 uniform CPU observations; report original objective, feasibility and retained quantum cost separately; common 30-bin histogram. |
| 23 | Select the best actually observed feasible assignment; a missing feasible observation stays None. |
| 26 | Separate 64-assignment exact reference after sampling replaces external Couenne. |

The original three inequalities are redundant for this dataset, but all three
remain explicit. Four primary assignments are feasible. The exact maximum is 3
at x=(0,0,1), computed only after optimization and sampling. The reference never
supplies parameters, probabilities or observations.

The frozen quantum encoding has three two-bit integer registers x0, x1, x2
and two one-bit slack outputs s0, s1. They are modeled outputs, not temporary
workspace; nonzero slack amplitudes must remain visible. The full joint domain
has 256 assignments. Zero workspace does not mean the two slack outputs are zero.

Retain the frozen quantum cost exactly:

```
10*(x0*x1+x0*x2+x1*x2) - (x0+2*x1+3*x2)
+10*(s0+x0+x1+x2-1)**2 +10*(s1+x0+x1+x2-1)**2
```

Each cost layer applies exp(-i*gamma*cost), including constant global phase.
Keep the source SUZUKI mixer calls for the two-bit registers and RX mixers on the
slack outputs. The independent Circuit implements their RX-per-bit equivalent.
Do not replace the joint quantum cost with a cost on primary marginals or claim
that its off-feasible penalties equal another independently chosen encoding.

Public native Nelder-Mead with simplex step 0.07 and initial parameters
(0.13,-0.27,0.19,0.22,0.31,-0.17) replaces the source private optimizer. Exact
local probabilities replace shot estimation during CVaR search. Every actual
evaluation synthesizes and executes. The 90-iteration setting is retained;
optimizer trajectories and iteration/evaluation semantics are not identical.
Final CPU sampling uses seed 2718 optimized and seed 31415 uniform. Report
actual evaluations, convergence and feasible sample counts, including poor
feasibility rates. A finite sampled optimum proves neither convergence nor
quantum advantage.

Named integer bit 0 is the least significant bit; physical axes are handled
separately. Readout requests x0, x1, x2, s0, s1 and validates their widths.
Acceptance must cover all 64 primary assignments, all 256 joint costs, absolute
phases at two nonzero gammas, two independent full complex states, named readout,
nonzero slack amplitudes, missing observations, bounds and resource refusals,
actual optimization and sampling, clean Notebook execution and actual Jupyter.

Classification: `single_device_fast_path`. No foreign quantum source loading,
external optimizer or exact-solver service is used.
