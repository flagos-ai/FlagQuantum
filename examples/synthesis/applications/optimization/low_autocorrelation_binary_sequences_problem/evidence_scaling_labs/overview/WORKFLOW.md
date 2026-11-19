# LABS circuit and sampling verification

The original cost sums squared correlations for lags zero through eleven.
It includes the constant zero-lag term and omits the final lag. Expansion into
Pauli products generates every nonzero support and retains the identity phase.
Independent direct-spin enumeration checks all 8,192 costs.

The bounded CX search builds three complete circuits with separately uncomputed
parities, sorted shared parities, and nearest-next shared parities. The named
U3 form carries a phase correction calculated from the generated Hamiltonian.
Measured CX count chooses the candidate, with depth as a tie-breaker.
Both nonzero bindings of all three candidates are checked at full complex
amplitude, including absolute phase. This is a minimum over the three
explicitly visited candidates.

The selected circuit reuses a symbolic six-angle template. Local public COBYLA
uses the explicit linear initial schedule and direct mean cost over the
executed probabilities. The callback enforces the source's 60-query setting.
The best evaluated parameter vector produces a separate 1,000-draw seeded
sample. The minimum and its degeneracy come from a separate exhaustive table
after sampling. The optimizer, estimator, and local sampling adapt the
unpinned hosted workflow.
