# Qmod tutorial part1

Category: tutorials / basics.

[Original Classiq source](https://raw.githubusercontent.com/Classiq/classiq-library/2bfd835e73b6882fcc515a725098b04766c238e7/tutorials/basic_tutorials/the_classiq_tutorial/Qmod_tutorial_part1.qmod) at `2bfd835e73b6882fcc515a725098b04766c238e7`.

## Implementations

| Form | Availability |
| --- | --- |
| Qmod original | [source.qmod](comparisons/source.qmod) |
| Native Python qfunc | Not yet provided for this exact original |
| Circuit comparison | See the validation source links below |

## Run

Run from the repository root with FlagQuantum installed.

```sh
python -m examples.synthesis.run_qmod 'tutorials/basic_tutorials/the_classiq_tutorial/Qmod_tutorial_part1.qmod'
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

Recorded resources: 5 physical qubits, 9 operations, depth 5.

Batch execution checks successful exact execution and basic result validity.
Linked tests and validators document their own input bindings, full-state
references, global-phase and workspace checks; inspect their individual scopes.

## Existing validation sources

- [test_synthesis_qmod_array_phase.py](../../../../../../../tests/integration/test_synthesis_qmod_array_phase.py)

Circuit-based validation source links: 1.

## Provenance

Original Git blob: `5507361161a5e78ebd4663dd08c7191d70b24d00`.

Original SHA-256: `2787ceb72a1199c2268bd0dc89b604c25a42da1bb539810518b2ba6eeacbfcc7`.

Approved SHA-256: `2787ceb72a1199c2268bd0dc89b604c25a42da1bb539810518b2ba6eeacbfcc7`.

[Classiq MIT license](../../../../../CLASSIQ_LICENSE.txt)
