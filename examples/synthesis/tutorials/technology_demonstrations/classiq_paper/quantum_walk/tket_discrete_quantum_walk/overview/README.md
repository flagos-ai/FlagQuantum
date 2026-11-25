# Native six-bit walk reference (tket source)

The original Notebook contains only commented code. This runnable migration uses
FlagQuantum exclusively; it does not execute or depend on the original SDK.

The source shift is modular increment. Coin=1 applies increment; coin=0 applies its inverse. The logical value is six-bit unsigned, with source array index zero as the low bit.

Target basis: CX/U3. U3 is the three-angle U gate. All explicit conversion rules preserve absolute phase.

The source calls Tket box decomposition, synthesis and an Aer backend. We report actual FlagQuantum CX/U3 costs instead of claiming the unspecified Aer/Tket basis or private pass equivalence.

Run from the repository root:

```bash
python -m examples.synthesis.tutorials.technology_demonstrations.classiq_paper.quantum_walk.tket_discrete_quantum_walk.overview.main
python -m examples.synthesis.tutorials.technology_demonstrations.classiq_paper.quantum_walk.tket_discrete_quantum_walk.overview.comparisons.circuit
```

Open [walkthrough.ipynb](walkthrough.ipynb). It visibly defines native qfuncs,
runs the source, compiles its target basis, reports actual resources and compares
an independent Circuit. See [WORKFLOW.md](WORKFLOW.md) for acceptance boundaries.
