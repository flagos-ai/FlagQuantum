# Exponential workflow acceptance

Fresh five-qubit probabilities proportional to exp(-0.1*n), exact local readout and all 32 complex amplitudes. This is fresh-output preparation, not a promise of the original in-place action on arbitrary entangled input.

Every original code cell is reviewed and hash-bound in provenance. Native Notebook
cells execute in order using captured Python qfunc definitions, synthesis and local
CPU execution. No quantum-source importer, cloud job or credentials are required.
The Python launcher reuses the existing native program and runs any explicitly
labeled coherent supplement as well.

Independent Circuit references check complete complex amplitudes, absolute phase,
named bit order and zero hidden work. Fresh-output and approximation boundaries
above remain binding. Notebook and frozen quantum-program acceptance are counted
separately; no distributed execution, gradient or resource-optimality claim is made.

```sh
python -m pytest tests/integration/test_synthesis_library_notebooks.py -q
```

The native fresh-state method uses a general probability preparation, while the
original uses a product of rotations. Their resources need not match. See the
[quantum-program scope](../WORKFLOW.md) for the preserved resource comparison.
