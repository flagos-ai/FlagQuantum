# ADAPT-QAOA

All twenty source stages, eight decision qubits, a three-layer ordinary comparator, all 74 candidate mixers in the original order, and adaptive growth until the source scaled gradient norm falls below 0.027. Each optimization allows 60 objective evaluations.

Run from the repository root with `python -m examples.synthesis.applications.optimization.adapt_qaoa.overview.main` or open the [Notebook](walkthrough.ipynb).

Native COBYLA replaces the hosted optimizer with the source ramp, an explicit query cap, and exact state expectations. Independent final draws use 1000 shots per variant with fixed local seeds. Public post-synthesis gate conversion measures the original CZ/RZ/RX/RY counts after verifying absolute amplitudes; the logical synthesis path preserves phase. A local 24-layer guard raises an error instead of treating exhaustion as convergence. This run does not reproduce historical hosted sampling or optimizer trajectories.

See [provenance](provenance.json).
