# Glued-tree quantum walks

All eight source stages: regenerate the six-qubit glued tree with seed 0 and its full 1,088-term Pauli expansion, use the original 200 cached twenty-qubit terms, execute both thirteen-point time grids with ten order-one Suzuki–Trotter repetitions, and draw 8,192 actual local samples per point.

From the repository root:

```bash
python -m examples.synthesis.algorithms.quantum_walks.glued_trees.overview.main
```

Open [the Notebook](walkthrough.ipynb) to compare exit probabilities across the two widths. The twenty-qubit run uses a full local state vector and can take several minutes. See [provenance](provenance.json) and [the retained data license](LICENSE-data.txt).

The six-qubit graph and coefficients are computed from scratch using a full Pauli basis, ignoring only numerical zeros below 1e-14. Equal-magnitude terms use the stable order of the native lexicographic Pauli enumeration; no claim of matching a historical SDK tie order. Each six-qubit Trotter slice uses two bounded synthesis calls, then the public Circuit API composes all ten slices before one coherent run. The twenty-qubit walk uses the unmodified ordered 200-term source cache subset, with source license and pins. Each point specializes the time at synthesis and samples with a seeded local stream. The figures show measured exit fractions and exact state probabilities.
