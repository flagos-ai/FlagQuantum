# classical shadow tomography

Category: algorithms / metrology.

[Original Classiq source](https://raw.githubusercontent.com/Classiq/classiq-library/2bfd835e73b6882fcc515a725098b04766c238e7/algorithms/metrology/classical_shadow_tomography.qmod) at `2bfd835e73b6882fcc515a725098b04766c238e7`.

## Implementations

| Form | Availability |
| --- | --- |
| Qmod original | [source.qmod](comparisons/source.qmod) |
| Native Python qfunc | Not yet provided for this exact original |
| Circuit comparison | See the validation source links below |

## Run

Run from the repository root with FlagQuantum installed.

```sh
python -m examples.synthesis.run_qmod 'algorithms/metrology/classical_shadow_tomography.qmod'
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

Recorded resources: 2 physical qubits, 7 operations, depth 6.

Batch execution checks successful exact execution and basic result validity.
Linked tests and validators document their own input bindings, full-state
references, global-phase and workspace checks; inspect their individual scopes.

## Existing validation sources

- [test_synthesis_qmod_bell.py](../../../../../../tests/integration/test_synthesis_qmod_bell.py)

Circuit-based validation source links: 1.

## Provenance

Original Git blob: `ec20f481553b87e3ab7e7712bb620bc2ea99b425`.

Original SHA-256: `47bb2be6b50c7e3753d5244b68035c7686a7350273a50f4a2e7c0494ce960464`.

Approved SHA-256: `47bb2be6b50c7e3753d5244b68035c7686a7350273a50f4a2e7c0494ce960464`.

[Classiq MIT license](../../../../CLASSIQ_LICENSE.txt)
