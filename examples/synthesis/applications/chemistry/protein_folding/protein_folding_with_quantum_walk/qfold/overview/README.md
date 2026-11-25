# QFold walk for alanylalanine

Run [the walkthrough](walkthrough.ipynb) or execute it from the repository root.

```bash
python -m examples.synthesis.applications.chemistry.protein_folding.protein_folding_with_quantum_walk.qfold.overview.main
```

The notebook starts from the original precomputed alanylalanine energy differences. It recomputes both Metropolis coin angles at inverse temperature 1, preserves their assignment in the source code, and synthesizes the original one-step walk. The run keeps all four logical and environment bits and draws 2,048 finite samples of the two dihedral-angle bits.

An independent Circuit verifies the full complex state, including the coin and move environment. The compiler's predicate work bit returns to zero. The original notebook ends with a fixed `01`-prefix lookup in the energy table, which is displayed separately from the sampled configurations. This workflow demonstrates one walk step and a lookup of archived energy differences.
