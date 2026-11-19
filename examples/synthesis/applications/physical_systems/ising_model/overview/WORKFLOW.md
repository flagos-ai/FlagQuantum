# Complete Ising workflow acceptance

Review all 11 source code cells and the surrounding formulation. Preserve the
periodic six-spin Hamiltonian, J=10, h=-20, five layers, quantile 0.7, actual
quantum solution table, uniform baseline, convergence trace and 30-bin weighted
energy histograms. The fixed Python energy replaces the source Pyomo model;
no generic modeling frontend is claimed. The source's private optimizer with
maxiter=100 is replaced by an explicit 100-evaluation local coordinate search.
The iteration contracts are not claimed to be identical.

Audit all 64 basis energies against independent periodic edge sums and the
frozen exported cost: source energy equals negative exported cost. Reuse that
quantum kernel with negated gammas without changing the frozen counterpart.
Compare full complex states for two nonzero parameter vectors, the uniform
baseline and actual best visited parameters against an independent Circuit
cost table and RX layers. No phase alignment is allowed. Check bit order,
zero workspace and resource-budget rejection.

Lower-tail cost must integrate fractional boundary mass and be independent of
input row order. Reject invalid quantiles, nonfinite or negative distributions,
duplicate or out-of-range configurations and invalid parameter vectors. Keep
all 100 evaluated costs, including rejected candidates; the trace may increase.
Reexecute the best parameters and verify the exact cost and distribution.
The optimizer receives no known optimum or desired ground state. Enumeration
is validation-only. A budget stop does not certify convergence.

Execute optimized and zero-angle uniform programs and sample 2048 local
outcomes from each with predetermined seeds 123 and 456. Retain all 64 rows,
including zeros. Check shot totals, spin interpretation, named readout,
reproducibility and actual best observed state. Check finite ground-state
probability separately from exact CVaR. Use 30 common histogram bins and the
actual sample weights.

Execute all four Notebook code cells and both CLI modules. Verify plots and
visually inspect the actual Jupyter PNGs. Saved source outputs remain empty.
Run all 216 original and six approved-revision replays. No private optimizer
identity, hardware, speedup, optimality or release certification is claimed.
