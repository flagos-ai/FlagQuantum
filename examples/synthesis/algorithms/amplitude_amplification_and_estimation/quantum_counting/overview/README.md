# Native quantum counting Notebook

Count two-bit integer pairs satisfying a+b<=2 with the original five-bit QPE
and a bounded count-only iterative estimator. The compiler owns allocation,
Grover composition and phase arithmetic. Read fractional phases and indicator
probabilities through public Program APIs.

```sh
python -m examples.synthesis.algorithms.amplitude_amplification_and_estimation.quantum_counting.overview.main
python -m examples.synthesis.algorithms.amplitude_amplification_and_estimation.quantum_counting.overview.comparisons.circuit
```

Open [the Notebook](walkthrough.ipynb) from the repository root. It retains the
100-point resolution calculation, QPE histogram, adaptive success counts,
confidence interval and solution-count assertions. Every finite sample is local:
2048 QPE draws and 256 new binomial shots per adaptive iteration, seed123.
Each visited native Grover state is computed once and its distribution cached.
No remote service, Classiq package or new production dependency is needed.

The source's requested epsilon=1/32 allows count error0.5, whereas its final
point assertion asks for error1/15. We retain the original run and explicitly
request epsilon=1/(16*15) for a second run. No seed selection or known-count
input repairs a result. The bounded Hoeffding estimator is example-local and
not an implementation-equivalence or speedup claim for Classiq IQAE. Its
confidence interval is probabilistic; QPE resolution is a separate quantity.
Dropped problem data is retained and need not be zero; workspace remains zero.

See [acceptance](WORKFLOW.md), [provenance](provenance.json) and the separate
[frozen native quantum program](../main.py).
