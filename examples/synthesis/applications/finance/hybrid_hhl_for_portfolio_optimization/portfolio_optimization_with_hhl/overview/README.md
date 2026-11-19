# Hybrid HHL portfolio allocation

Run the [walkthrough](walkthrough.ipynb) in a FlagQuantum Python environment with
NumPy and Matplotlib. From the repository root, the same workflow runs with:

```sh
python -m examples.synthesis.applications.finance.hybrid_hhl_for_portfolio_optimization.portfolio_optimization_with_hhl.overview.main
```

The original two-asset portfolio gives a four-dimensional constrained linear
system. The example rebuilds that system and uses a four-bit phase-estimation
program to collect 2,048 samples. The observed rows select two reciprocal
rotations for a three-bit HHL phase counter. The basic HHL program uses all 16
four-bit rotations. Both approaches use a coherent SWAP test against the
normalized direct solution, and the width profile reuses the depth profile's
counts as in the source. All three synthesis calls use a 12-qubit ceiling.
The programs can choose the same circuit for the depth and width objectives.

The native program runs in the local simulator. Finite-shot histograms are
seeded draws from its exact named-register probabilities. The resource table
counts gates in a separate CX/U3 conversion with a verified global-phase-aligned
state. The absolute-phase test applies to the native circuits. The decomposition
uses the local compiler's rules, so its depth and gate counts depend on those
rules.
