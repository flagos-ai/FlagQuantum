# fermi hubbard 1D

Category: applications / physical_systems.

[Original Classiq source](https://raw.githubusercontent.com/Classiq/classiq-library/2bfd835e73b6882fcc515a725098b04766c238e7/applications/physical_systems/fermi_hubbard_model_1D/fermi_hubbard_1D.qmod) at `2bfd835e73b6882fcc515a725098b04766c238e7`.

## Implementations

| Form | Availability |
| --- | --- |
| Qmod original | [source.qmod](comparisons/source.qmod) |
| Native Python qfunc | Not yet provided for this exact original |
| Circuit comparison | See the validation source links below |

## Run

Run from the repository root with FlagQuantum installed.

```sh
python -m examples.synthesis.run_qmod 'applications/physical_systems/fermi_hubbard_model_1D/fermi_hubbard_1D.qmod'
```

The shared runner binds the recorded classical inputs and selects the
recorded exact CPU execution profile. It prints measured resources and
execution validation, using the same audit implementation as corpus acceptance.

Classical bindings used by this entry:

```json
{
  "num_iter": 2
}
```

## Verification

Recorded implementation commit: `7675271cb758cd700aa31962b004d1f87b479518`.

Original result: `execution_smoke_passed`. Approved result: `execution_smoke_passed`.

Recorded resources: 8 physical qubits, 268 operations, depth 109.

Batch execution checks successful exact execution and basic result validity.
Linked tests and validators document their own input bindings, full-state
references, global-phase and workspace checks; inspect their individual scopes.

## Existing validation sources

- [test_synthesis_qmod_integer_composition.py](../../../../../../../tests/integration/test_synthesis_qmod_integer_composition.py)
- [validate_qmod_integer_composition.py](../../../../../../../tools/validate_qmod_integer_composition.py)

Circuit-based validation source links: 2.

## Provenance

Original Git blob: `b1fbf1664f2732776fc058f6ba9063155888757d`.

Original SHA-256: `7cf818c5d745cfb719f195242aa04f0eadaa55ef82b55bb7ae889c4d92e351e8`.

Approved SHA-256: `7cf818c5d745cfb719f195242aa04f0eadaa55ef82b55bb7ae889c4d92e351e8`.

[Classiq MIT license](../../../../../CLASSIQ_LICENSE.txt)
