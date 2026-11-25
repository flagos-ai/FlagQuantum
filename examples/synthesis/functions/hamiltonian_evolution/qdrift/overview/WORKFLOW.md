# Qdrift workflow acceptance

Original two-qubit XY+0.5Z Hamiltonian, evolution coefficient 2.0 and all 5 samples. Fixed seed-zero sampled sequence is checked against an independent Circuit; not equality to exact Hamiltonian evolution.

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
