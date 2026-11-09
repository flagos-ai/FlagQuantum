# suzuki trotter

Category: tutorials / evolution.

[Original Classiq source](https://raw.githubusercontent.com/Classiq/classiq-library/2bfd835e73b6882fcc515a725098b04766c238e7/functions/qmod_library_reference/qmod_core_library/hamiltonian_evolution/suzuki_trotter/suzuki_trotter.qmod) at `2bfd835e73b6882fcc515a725098b04766c238e7`.

## Implementations

| Form | Availability |
| --- | --- |
| Qmod original | [source.qmod](comparisons/source.qmod) |
| Native Python qfunc | Not yet provided for this exact original |
| Circuit comparison | See the validation source links below |

## Run

Run from the repository root with FlagQuantum installed.

```sh
python -m examples.synthesis.run_qmod 'functions/qmod_library_reference/qmod_core_library/hamiltonian_evolution/suzuki_trotter/suzuki_trotter.qmod'
```

The shared runner binds the recorded classical inputs and selects the
recorded exact CPU execution profile. It prints measured resources and
execution validation, using the same audit implementation as corpus acceptance.

Classical bindings used by this entry:

```json
{
  "x": 0.37
}
```

## Verification

Recorded implementation commit: `7675271cb758cd700aa31962b004d1f87b479518`.

Original result: `execution_smoke_passed`. Approved result: `execution_smoke_passed`.

Recorded resources: 3 physical qubits, 16 operations, depth 14.

Batch execution checks successful exact execution and basic result validity.
Linked tests and validators document their own input bindings, full-state
references, global-phase and workspace checks; inspect their individual scopes.

## Existing validation sources

- [test_synthesis_qmod_suzuki_trotter.py](../../../../../../../../../../tests/integration/test_synthesis_qmod_suzuki_trotter.py)

Circuit-based validation source links: 1.

## Provenance

Original Git blob: `82b981254ad43471828386aeb8f4ed1546bc6a90`.

Original SHA-256: `49625dc957773acbc0722b77c25e6af997c815694bf6b73301cd964d8906ab56`.

Approved SHA-256: `49625dc957773acbc0722b77c25e6af997c815694bf6b73301cd964d8906ab56`.

[Classiq MIT license](../../../../../../../../CLASSIQ_LICENSE.txt)
