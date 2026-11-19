# Negation workflow acceptance

The source has two code cells: a three-bit signed uniform input with b=-a,
then synthesis/sampling. All eight input values -4 through 3 are retained. The
result must hold 4 for input -4, requiring four signed output bits; the compiler
infers this width rather than wrapping -(-4) back to -4.

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
