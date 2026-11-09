# quantum autoencoder

Category: algorithms / machine_learning.

[Original Classiq source](https://raw.githubusercontent.com/Classiq/classiq-library/2bfd835e73b6882fcc515a725098b04766c238e7/algorithms/QML/quantum_autoencoder/quantum_autoencoder.qmod) at `2bfd835e73b6882fcc515a725098b04766c238e7`.

## Implementations

| Form | Availability |
| --- | --- |
| Qmod original | [source.qmod](comparisons/source.qmod) |
| Native Python qfunc | [Corresponding native program](../../../../../applications/machine_learning/autoencoder/main.py) |
| Circuit comparison | [Runnable three-form comparison](../../../../../applications/machine_learning/autoencoder/comparisons/circuit.py) |

## Run

Run from the repository root with FlagQuantum installed.

```sh
python -m examples.synthesis.run_qmod 'algorithms/QML/quantum_autoencoder/quantum_autoencoder.qmod'
```

The shared runner binds the recorded classical inputs and selects the
recorded exact CPU execution profile. It prints measured resources and
execution validation, using the same audit implementation as corpus acceptance.

Classical bindings used by this entry:

```json
{
  "input_data": [
    0.19,
    -0.31,
    0.43,
    -0.59
  ],
  "w": [
    0.11,
    -0.23,
    0.37,
    -0.41,
    0.53,
    -0.67,
    0.71,
    -0.83,
    0.97,
    -1.09
  ]
}
```

## Verification

Recorded implementation commit: `7675271cb758cd700aa31962b004d1f87b479518`.

Original result: `execution_smoke_passed`. Approved result: `execution_smoke_passed`.

Recorded resources: 4 physical qubits, 16 operations, depth 8.

Batch execution checks successful exact execution and basic result validity.
Linked tests and validators document their own input bindings, full-state
references, global-phase and workspace checks; inspect their individual scopes.

Run the native program and its three-form comparison:

```sh
python -m examples.synthesis.applications.machine_learning.autoencoder.main
python -m examples.synthesis.applications.machine_learning.autoencoder.comparisons.circuit
```

## Existing validation sources

- [test_synthesis_qmod_binding_scopes.py](../../../../../../../tests/integration/test_synthesis_qmod_binding_scopes.py)
- [validate_qmod_binding_scopes.py](../../../../../../../tools/validate_qmod_binding_scopes.py)

Circuit-based validation source links: 2.

## Provenance

Original Git blob: `96bebed5db9119ee2ca6e2cb87b022cc4bdaf17f`.

Original SHA-256: `b718ab54e71f1fd79d440342121af585522d8e2918f1a98b24a751b808f2d3c8`.

Approved SHA-256: `b718ab54e71f1fd79d440342121af585522d8e2918f1a98b24a751b808f2d3c8`.

[Classiq MIT license](../../../../../CLASSIQ_LICENSE.txt)
