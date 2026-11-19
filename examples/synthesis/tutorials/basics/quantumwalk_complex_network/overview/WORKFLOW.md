# Workflow acceptance

The original quantum program is already accepted independently. This workflow adds
its complete host-side graph construction, readout and three plots. Full complex
position-plus-coin amplitudes match an independent Householder/block-reflection
Circuit for rounds0..3. Absolute phase is retained. The public position marginal
agrees with the independent partial trace; hidden coin entropy is permitted.
One actual compiler work bit is checked separately for zero amplitude. `drop(y)`
does not reset, measure or make the coin a reusable clean bit.

Both archived graph sizes retain all times1..4,10000 counts per hardware row and
source-rounded ideal probabilities. No source hardware observations are replaced
by local output. Plot tests verify the five-panel series against the declared
hardware data, preventing the source's swapped-label bug from returning.
Historical four-node ideal rows are compared with local marginals within1e-8.
The eight-node historical arrays are not asserted to be independently rerun.

All original code cells, including commented hardware examples, are reviewed and
hash-bound. The native Notebook has four code cells and no saved output. Actual
Jupyter and CLI execution complement test execution; figures are visually checked.
Numerical checks do not depend on Matplotlib. Plot checks skip only if that
optional package is unavailable. NetworkX is an optional workflow dependency.

```sh
python -m pytest tests/integration/test_synthesis_network_walk_notebook.py -q
```
