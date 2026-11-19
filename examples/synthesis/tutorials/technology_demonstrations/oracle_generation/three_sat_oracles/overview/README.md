# Generating 3-SAT phase oracles

Regenerate all 23 original problem sizes using seed 128. Each variable contributes
one three-literal clause. The oracle applies a phase of minus one on satisfying
assignments and restores every data and work bit.

```bash
python -m examples.synthesis.tutorials.technology_demonstrations.oracle_generation.three_sat_oracles.overview.main
```

Open [the Notebook](walkthrough.ipynb) to inspect the generated Python condition,
the small coherent reference and the measured construction sweep. The timing
plot identifies the local construction routes and the original Notebook's
recorded Qiskit values. Local times vary by machine and run.
