# Six-residue protein folding with native QAOA

Build the original `ABCDEF` objective on six direction bits and one contact
bit. The five QAOA layers synthesize the cost from commuting Pauli parities and
preserve its identity phase. All 128 classical costs agree with the existing
independent folding reference.

Run from the repository root with
`python -m examples.synthesis.applications.chemistry.protein_folding.protein_folding_with_qaoa.protein_folding.overview.main`.
The [Notebook](walkthrough.ipynb) walks through the original model, optimization,
sampled folds, and exact classical comparison.

The local COBYLA run uses the grouped pi-ramp initialization, the lower 70%
sample mean, at most 70 objective evaluations, and reproducible independent
seeds. Each evaluation and each final comparison uses 2,048 samples. The shot
count matches the upstream displayed frequencies. The final readout uses the
last evaluated binding. Exhaustive comparison over all 128 feasible assignments
replaces the external classical solver. It does not change the observed sample.

[Source provenance](provenance.json).
