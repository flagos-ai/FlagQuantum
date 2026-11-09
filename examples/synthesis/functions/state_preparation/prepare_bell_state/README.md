# prepare bell state

This example uses native FlagQuantum Python quantum functions.

```sh
python -m examples.synthesis.functions.state_preparation.prepare_bell_state.main
python -m examples.synthesis.functions.state_preparation.prepare_bell_state.comparisons.circuit
```

The Circuit comparison checks complete complex amplitudes, absolute phase,
declared register order and hidden workspace. It does not call a source importer.

The quantum program and complete local Notebook workflow are validated.
The workflow scope and replacements for external services
are documented in [the workflow guide](WORKFLOW.md).

Reference identity: `functions/qmod_library_reference/classiq_open_library/special_state_preparations/prepare_bell_state.qmod` at `2bfd835e73b6882fcc515a725098b04766c238e7`.

Provenance is recorded in `provenance.json`; original sources and importer
regressions live only in test fixtures. The shared third-party license applies.

[Third-party license](../../../CLASSIQ_LICENSE.txt)
