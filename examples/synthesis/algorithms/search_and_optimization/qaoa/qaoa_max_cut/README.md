# qaoa max cut

This example uses native FlagQuantum Python quantum functions.

```sh
python -m examples.synthesis.algorithms.search_and_optimization.qaoa.qaoa_max_cut.main
python -m examples.synthesis.algorithms.search_and_optimization.qaoa.qaoa_max_cut.comparisons.circuit
```

The Circuit comparison checks complete complex amplitudes, absolute phase,
declared register order and hidden workspace. It does not call a source importer.

The validated scope is the quantum program. A related notebook may also include
training, data preparation or external execution that requires separate migration.
See the [quantum-program guide](WORKFLOW.md).

Reference identity: `algorithms/search_and_optimization/QAOA/qaoa_max_cut.qmod` at `2bfd835e73b6882fcc515a725098b04766c238e7`.

Provenance is recorded in `provenance.json`; original sources and importer
regressions live only in test fixtures. The shared third-party license applies.

[Third-party license](../../../../CLASSIQ_LICENSE.txt)
