# Six-bit periodic quantum walk: native compilation workflow

This Notebook preserves the source's six position bits, coin, conditional
increment/decrement, maximum width of twelve and CX/U hardware basis objective.
It compiles two semantically equivalent high-level programs, converts both to
CX/U3, reports actual width, depth and gate counts, and selects the visited
candidate with fewer CX gates (ties: depth, width, name).

U3 is FlagQuantum's name for the three-angle U gate. The explicit, immutable
conversion-rule extension preserves absolute phase and leaves compiler defaults
unchanged. This is bounded selection between two candidates, not a claim of
globally optimal compilation or equivalence to a private optimizer.

Run from the repository root:

```bash
python -m examples.synthesis.tutorials.technology_demonstrations.classiq_paper.quantum_walk.classiq_discrete_quantum_walk.overview.main
python -m examples.synthesis.tutorials.technology_demonstrations.classiq_paper.quantum_walk.classiq_discrete_quantum_walk.overview.comparisons.circuit
```

Open [walkthrough.ipynb](walkthrough.ipynb) for native `qfunc` definitions,
compilation, cost selection, named results and independent comparison.
See [WORKFLOW.md](WORKFLOW.md) for source coverage and acceptance boundaries.
