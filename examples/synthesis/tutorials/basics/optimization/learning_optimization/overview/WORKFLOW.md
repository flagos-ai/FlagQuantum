# Source workflow mapping

Pinned source: `tutorials/basic_tutorials/optimization/learning_optimization.ipynb`
at the commit and code-cell hashes in `provenance.json`.

| Source cells | Native workflow |
| --- | --- |
| 2, 4 | Replace foreign modeling imports with native qfunc and the public Nelder-Mead optimizer. |
| 6, 8, 10, 12, 14 | Keep the two nonnegative integer domains 0..3, cost 3*x+2*y and inequality 3*x+y>=2. |
| 16 | Keep one layer and penalty20; declare the explicit native positive-part-squared encoding. |
| 18, 21, 28 | Native synthesis, resource inspection and actual local execution. |
| 23, 25, 26 | Actual bounded optimizer evaluations, local samples, cost sorting and best feasible observation. |

The source's Pyomo converter chooses a private slack encoding and optimizer.
This workflow uses `C=3*x+2*y+20*max(0,2-3*x-y)^2`, phase `exp(-i*gamma*C)`
and `RX(2*beta)` on each bit. It preserves the optimization problem, domain,
penalty and layer count, but does not claim identical circuits, landscape or
optimizer trajectories. The frozen source Qmod and its existing eight-bit
native counterpart retain their distinct slack weights; neither is replaced
by this four-bit Notebook reconstruction.

The existing public Nelder-Mead optimizer starts at `(0.19,0.37)`, uses step
0.07 and at most32 iterations. Every evaluation uses `synthesize`, `fq.run`
and named probabilities. Sampling uses1000 local CPU multinomial draws with
seed2718. Report the convergence flag, penalized expectation, feasible count
and best observed unpenalized cost separately. No foreign SDK or cloud runs.

Acceptance compares full complex statevectors at two nonzero angle pairs,
all16 assignments and four absolute cost phases, independent feasible-optimum
enumeration, nonconvergence reporting, seeded samples and budget refusals.
The independent Circuit uses a 4096-byte cost diagonal solely for verification.
No dense matrix enters native synthesis. Runtime path: single_device_fast_path.
