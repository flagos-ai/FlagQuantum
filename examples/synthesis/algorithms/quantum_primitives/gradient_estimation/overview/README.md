# Native quantum gradient estimation

Run the complete source-sized workflow:

```bash
python -m examples.synthesis.algorithms.quantum_primitives.gradient_estimation.overview.main
```

[walkthrough.ipynb](walkthrough.ipynb) includes all 14 original code stages:
auxiliary phase kickback, direct linear/quadratic phases, signed gradient
readout, all 17 m values and nine l values, and the two-dimensional gradient.
Every source program is compared with an independent complete Circuit state.
The workflow makes 31 batches of 1000 actual local draws and generates all ten
original analysis plots. No provider account is required.

[WORKFLOW.md](WORKFLOW.md) records translation choices and exact support limits.
