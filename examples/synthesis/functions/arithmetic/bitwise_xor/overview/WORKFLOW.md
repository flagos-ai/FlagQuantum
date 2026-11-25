# Bitwise XOR workflow acceptance

The upstream Notebook has four code cells: the two-register quantum definition,
its synthesis/sampling, the integer-mask definition, and its synthesis/sampling.
All four are covered by the native [Notebook](walkthrough.ipynb); no upstream code
cell is skipped. The native Python launcher reuses the existing qfunc examples.

Original widths and signs are retained: signed five-bit a=4 and unsigned
three-bit b=5 produce res=1. The second example preserves a=4 and mask3,
producing7. Numeric named reads retain signed interpretation. CPU execution
replaces cloud execution;1000seeded samples are drawn from executed exact
probabilities. No hardware, noisy-shot or speedup claim is made.

Acceptance executes every native code cell in source order in a fresh namespace
with source caching for qfunc inspection, without host AST rewriting. This
Python-cell harness does not certify Jupyter rendering or a particular kernel
installation. Both complete complex states are checked against independent
Circuit references, including absolute phase and hidden work. The two source
quantum-program counterparts were already included in216; this adds one full
Notebook workflow rather than two more quantum programs.

```sh
python -m pytest tests/integration/test_synthesis_bitwise_notebooks.py -q
```
