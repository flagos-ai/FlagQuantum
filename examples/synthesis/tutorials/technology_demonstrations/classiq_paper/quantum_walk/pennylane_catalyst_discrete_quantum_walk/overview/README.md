# Native six-bit walk reference (catalyst source)

The original Notebook contains only commented code. This runnable migration uses
FlagQuantum exclusively; it does not execute or depend on the original SDK.

Preserve the original loop indentation: X(x[0]) occurs inside all five MCX iterations. This is a reversible permutation, but differs from modular increment on 32 of 64 positions. Do not silently replace it with arithmetic increment.

Target basis: CX/RZ/RY/RX. Exact logical phase uses one final RZ on an existing clean zero work qubit. Its cost is included; the transform is an isometry with zero input/output work, not an identity for dirty work.

The source requests CX/RZ/RY/RX then counts CNOT text in a jitted MLIR string. We count actual executable CX instructions instead. No qjit, Lightning runtime or MLIR-text timing equivalence is claimed.

Run from the repository root:

```bash
python -m examples.synthesis.tutorials.technology_demonstrations.classiq_paper.quantum_walk.pennylane_catalyst_discrete_quantum_walk.overview.main
python -m examples.synthesis.tutorials.technology_demonstrations.classiq_paper.quantum_walk.pennylane_catalyst_discrete_quantum_walk.overview.comparisons.circuit
```

Open [walkthrough.ipynb](walkthrough.ipynb). It visibly defines native qfuncs,
runs the source, compiles its target basis, reports actual resources and compares
an independent Circuit. See [WORKFLOW.md](WORKFLOW.md) for acceptance boundaries.
