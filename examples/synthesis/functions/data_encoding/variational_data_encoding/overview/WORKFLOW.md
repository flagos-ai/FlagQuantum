# Encoding workflow acceptance

Retain both four-feature angle and seven-feature Bloch stages, including the odd Bloch tail. Introduce two explicitly labeled nonzero demonstration bindings per stage and execute locally. The original declares symbolic inputs; these concrete values are not attributed to the source.

Every original code cell is reviewed and hash-bound in provenance. Native Notebook
cells execute in order and use captured Python qfunc definitions, synthesis,
local CPU execution and named outputs. The launcher reuses existing native
programs; no source-language importer or remote credentials are required.

Independent Circuit references check complete complex amplitudes, absolute phase,
named bit order and zero hidden work. Numeric truth tables, declared domains,
sampling totals and plot contents are checked separately where applicable.
Classical plotting uses the optional Matplotlib dependency; tests requiring it
are distinct from the other three workflows. Real Jupyter execution also checks
the complete plotting workflow when present.

Notebook and frozen quantum-program acceptance remain separate. There is no
hardware, distributed execution, resource-optimality or gradient claim.

```sh
python -m pytest tests/integration/test_synthesis_search_encoding_notebooks.py -q
```
