# quantum thermal state preparation

This example uses native FlagQuantum Python quantum functions.

```sh
python -m examples.synthesis.algorithms.quantum_state_preparation.gibbs.quantum_thermal_state_preparation.main
python -m examples.synthesis.algorithms.quantum_state_preparation.gibbs.quantum_thermal_state_preparation.comparisons.circuit
```

The Circuit comparison checks the complete density matrix, register order
and reset/hidden environment. Separate unitary-block tests check complex
amplitudes and absolute phase. Neither comparison calls a source importer.

The validated scope is the quantum program. A related notebook may also include
training, data preparation or external execution that requires separate migration.
See the [quantum-program guide](WORKFLOW.md).

Reference identity: `algorithms/quantum_state_preparation/gibbs/quantum_thermal_state_preparation.qmod` at `2bfd835e73b6882fcc515a725098b04766c238e7`.

Provenance is recorded in `provenance.json`; original sources and importer
regressions live only in test fixtures. The shared third-party license applies.

[Third-party license](../../../../CLASSIQ_LICENSE.txt)
