# Comparator workflow acceptance

The source has four code cells: signed-five-bit/unsigned-three-bit equality
and its synthesis, then unsigned-three-bit <=2 over uniform input and its
synthesis/sampling. Native equality additionally executes and samples the
zero/zero/one outcome. The threshold retains all eight coherent inputs; its
result is one for 0, 1, 2 and zero otherwise.

Every source code cell is reviewed and hash-bound in provenance. All native
Notebook code cells execute in order in both a source-cached Python test
namespace and a real Jupyter kernel. The launcher reuses the existing native
qfunc definitions; it does not translate source text or invoke a source importer.
Independent Circuit references compare every complex amplitude, absolute phase,
named bit order and zero hidden work. Named numeric readout preserves signed
values without manual two-complement decoding.

The cloud sampler is replaced by 1000 seeded samples from locally executed exact
probabilities. Sampling does not measure mid-program, certify noisy hardware or
verify phase by itself. Quantum-only coverage remains 216; this workflow adds
one separately tracked Notebook. The distributed/gradient and general Python
language contracts are unchanged.

```sh
python -m pytest tests/integration/test_synthesis_comparison_notebooks.py -q
```
