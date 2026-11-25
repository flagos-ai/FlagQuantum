# Quantum autocallable option pricing

This walkthrough regenerates a three-year autocallable from its original
financial assumptions, including four Gaussian bins per period, two fractional
bits, two prioritized early calls, and the downside barrier. It enumerates all
64 paths, then executes the native coherent preparation and one Grover iterate.

```bash
python -m examples.synthesis.applications.finance.autocallable_options.quantum_autocallable_option_pricing.overview.main
```

Open [the Notebook](walkthrough.ipynb) for the derivation and execution. The
marker probability of the preparation decodes to the discounted expected payoff.
The Grover iterate is checked separately against its complete reference state.
The source leaves its long adaptive IQAE run disabled. The walkthrough keeps
that setting and labels the source-reported estimate and confidence interval as
historical.
