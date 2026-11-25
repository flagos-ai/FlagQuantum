# QAOA on a seven-vertex maximum-clique problem

The [walkthrough](walkthrough.ipynb) generates the source graph from seed 79 using Python's random generator,
evaluates its squared missing-edge penalty at all 128 words and builds the cost
Hamiltonian from the resulting 61 nonzero coefficients. The native Python
qfunc applies ten alternating cost and mixer layers.

Each objective evaluation draws 10,000 outcomes from the native program's
named probabilities. FlagQuantum COBYLA uses the mean energy and the
norm-scaled initial schedule. The workflow allows up to 400 trust-region
iterations and caps actual measured queries at 400. It samples the best
evaluated point once more to read individual words and reports whether its
selected word is a clique. The classical exact energy range is diagnostic.

The saved plot compares the freshly sampled FlagQuantum trace with the
upstream's 218-point historical Qiskit series. The FlagQuantum depth and CX
counts use a separate CX/U3 conversion whose state is checked up to global
phase. The source Qiskit depth and CX count come from the original Notebook.

Run from the repository root with FlagQuantum and Matplotlib.

```sh
python -m examples.synthesis.tutorials.technology_demonstrations.qaoa.qaoa_demonstration.overview.main
```
