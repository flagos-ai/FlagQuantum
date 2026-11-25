# Native quantum Monte Carlo integration

Retain the source probability distribution p(x)=x/28 for x=0..7 and the
function sin²(0.25x+0.2). Build the probability/indicator loader, a complete
Grover operator and three-bit signed fractional QPE using native qfunc
composition. No source importer, external service or Classiq SDK is used.

```sh
python -m examples.synthesis.algorithms.amplitude_amplification_and_estimation.qmc_user_defined.overview.main
python -m examples.synthesis.algorithms.amplitude_amplification_and_estimation.qmc_user_defined.overview.comparisons.circuit
```

Open [the complete Notebook](walkthrough.ipynb) from the repository root. It
shows all three stages, reports actual resources, samples2048 local phase
outcomes with seed123, plots the signed phase histogram and compares the
estimated amplitude with the exact classical expectation at source tolerance0.1.
Probability loading and inverse composition use the full loader unitary.

The source computes full-precision i/28 probabilities. The separately frozen
[quantum counterpart](../main.py) retains rounded Qmod decimal constants;
we do not change those original constants or relabel this as an identical
frozen fullstate. This complete workflow validates source-precision states
against independent matrices with identical full-precision inputs.

Dropped data remains physically retained. It is not workspace required to
return to zero. All three native programs have zero workspace; the QPE
program has seven physical qubits. Sampling is a finite classical draw from
actual local probabilities, not hardware execution or evidence of speedup.
QPE resolution is not a statistical confidence interval.

See [acceptance](WORKFLOW.md) and [source provenance](provenance.json).
