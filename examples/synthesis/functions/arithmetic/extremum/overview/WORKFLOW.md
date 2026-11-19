# Extremum workflow acceptance

First compare fixed a=4 with all eight coherent three-bit values b and compute min(a,b). Then compute max(3.5,a) over all eight coherent a values, including values on either side of the threshold.

Both original examples remain in source order. Every original code cell is hash-bound
in provenance. All five native code cells execute in a source-cached namespace and
in a real Jupyter kernel. Independent Circuit references compare the full complex
state, absolute phase, named bit order and zero hidden workspace. The Python
launcher reuses the existing native definitions; it does not import Qmod.

Each stage reports exact named probabilities, resources and 1000 seeded host samples.
These samples replace the cloud sampler; they do not certify hardware or phase.
This adds one separately tracked Notebook workflow; quantum-program coverage remains
216. General Python, distributed execution and gradient contracts are unchanged.

```sh
python -m pytest tests/integration/test_synthesis_arithmetic_notebooks.py -q
```
