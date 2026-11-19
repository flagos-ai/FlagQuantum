# Native six-data-bit QSVT compilation

This workflow keeps the source's degree three, four exact real phases, six data
bits, three block bits, QSVT auxiliary output and eleven-physical-bit ceiling.
Native qfuncs compose initialization, modular arithmetic on logical views, a
reflection, diagonal phase expressions and an adjoint. No physical wire mapping
or bitstring decoding is needed in user code.

The actual executable targets CX/U3 (U3 names the three-angle U gate). Explicit
caller-owned exact rules extend the existing walk rules with a two-U3 RZ
identity that preserves absolute phase without another work bit. Shared compiler
defaults remain unchanged. Actual CX count, depth, width, synthesis time,
conversion time and total local workflow time are reported. The source CX
objective is measured in this basis; a globally optimal result, the private
compiler's search/custom passes and its timing are not reproduced or claimed.

```bash
python -m examples.synthesis.tutorials.technology_demonstrations.classiq_paper.qsvt.classiq_qsvt.overview.main
python -m examples.synthesis.tutorials.technology_demonstrations.classiq_paper.qsvt.classiq_qsvt.overview.comparisons.circuit
```

Open [walkthrough.ipynb](walkthrough.ipynb) for an inline native qfunc, measured
compilation and named auxiliary-output probabilities. Source Notebook outputs
are empty. See [WORKFLOW.md](WORKFLOW.md) for mapping and acceptance.
