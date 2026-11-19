# Native periodic Ising optimization

Minimize the six-spin periodic energy with J=10, h=-20 and five QAOA layers.
This complete workflow retains CVaR quantile 0.7, 100 objective evaluations,
actual sampling, ranked solutions, convergence and optimized/uniform histograms.

```sh
python -m examples.synthesis.applications.physical_systems.ising_model.overview.main
python -m examples.synthesis.applications.physical_systems.ising_model.overview.comparisons.circuit
```

Open [the Notebook](walkthrough.ipynb) from the repository root. Its native
qfunc combines uniform preparation, cost phases and RX mixing without physical
wires or bitstrings. Public named probabilities provide integer configurations,
interpreted as six spins: 0 maps to -1 and 1 maps to +1. The explicit Python
energy replaces the source's fixed Pyomo formulation; this is not a general
Pyomo importer. No Classiq SDK or new production dependency is needed.

The source energy is the negative of the frozen exported Qmod cost on all 64
states. Reuse the existing cost kernel with negated gamma parameters and verify
source-energy states independently. Keep the [frozen counterpart](../main.py)
and its resource/full-state evidence unchanged. An explicit bounded coordinate
search replaces an unspecified private optimizer. Its initial angles, steps,
100-evaluation budget and fractional-tail integration are visible in
[workflow.py](workflow.py); no optimizer-equivalence claim is made.

Every objective evaluation synthesizes and runs the native quantum program.
Optimization uses exact local probabilities. Final tables use 2048 local
samples each from optimized and independently executed uniform programs, with
fixed seeds 123 and 456. Local sampling is not hardware execution. The optimizer
receives neither a known optimum nor a ground-state target. Classical
enumeration is used only to validate and report results.

A minimum CVaR value does not mean a pure ground state: report ground probability
separately. Termination after 100 evaluations means budget exhaustion, without
convergence certification. No hardware, speedup or global optimizer guarantee is
made. See [acceptance](WORKFLOW.md) and [provenance](provenance.json).
