# Linear workflow acceptance

Six-bit control and three Y targets with retained slopes/offsets, original zero input and supplemental coherent phased control. All512complex amplitudes and zero work are checked.

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
