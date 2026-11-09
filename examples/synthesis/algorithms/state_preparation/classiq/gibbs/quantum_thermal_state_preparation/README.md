# quantum thermal state preparation

Category: algorithms / state_preparation.

[Original Classiq source](https://raw.githubusercontent.com/Classiq/classiq-library/2bfd835e73b6882fcc515a725098b04766c238e7/algorithms/quantum_state_preparation/gibbs/quantum_thermal_state_preparation.qmod) at `2bfd835e73b6882fcc515a725098b04766c238e7`.

## Implementations

| Form | Availability |
| --- | --- |
| Qmod original | [source.qmod](comparisons/source.qmod) |
| Native Python qfunc | Not yet provided for this exact original |
| Circuit comparison | See the validation source links below |

## Run

Run from the repository root with FlagQuantum installed.

```sh
python -m examples.synthesis.run_qmod 'algorithms/quantum_state_preparation/gibbs/quantum_thermal_state_preparation.qmod'
```

The shared runner binds the recorded classical inputs and selects the
recorded exact CPU execution profile. It prints measured resources and
execution validation, using the same audit implementation as corpus acceptance.

Classical bindings used by this entry:

```json
null
```

## Verification

Recorded implementation commit: `7675271cb758cd700aa31962b004d1f87b479518`.

Original result: `execution_smoke_passed`. Approved result: `execution_smoke_passed`.

Recorded resources: 11 physical qubits, 22200 operations, depth 19901.

Batch execution checks successful exact execution and basic result validity.
Linked tests and validators document their own input bindings, full-state
references, global-phase and workspace checks; inspect their individual scopes.

## Existing validation sources

- [test_synthesis_qmod_thermal_reset.py](../../../../../../../tests/integration/test_synthesis_qmod_thermal_reset.py)

Circuit-based validation source links: 1.

## Provenance

Original Git blob: `a311f41d78c7a7fd384aa989b8745fedf9edda8b`.

Original SHA-256: `2a966c74216e513a8e492982791cdb0d85a8273fa89dd852ce908de88c68ccb9`.

Approved SHA-256: `2a966c74216e513a8e492982791cdb0d85a8273fa89dd852ce908de88c68ccb9`.

[Classiq MIT license](../../../../../CLASSIQ_LICENSE.txt)
